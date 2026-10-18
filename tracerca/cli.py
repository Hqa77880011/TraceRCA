import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import gammaln


def paths(graph, start, target, depth):
    if start == target:
        yield []
    if depth == 0:
        return
    for edge, relation in enumerate(graph['edges']):
        if relation['u'] == start:
            for rest in paths(graph, relation['v'], target, depth - 1):
                nodes = [graph['edges'][e]['v'] for e in rest]
                if start not in nodes:
                    yield [edge] + rest


def fit(args):
    graph = json.loads(Path(args.graph).read_text())
    episodes = json.loads(Path(args.data).read_text())
    n = len(graph['nodes'])
    kinds = sorted(set(e['type'] for e in graph['edges']))
    type_index = torch.tensor([kinds.index(e['type']) for e in graph['edges']])
    initial = np.array([graph['kernels'][kind]['mean'] for kind in kinds])
    shape = torch.tensor([graph['kernels'][e['type']]['shape'] for e in graph['edges']], dtype=torch.float64)
    log_type = torch.tensor(np.log(initial), dtype=torch.float64, requires_grad=True)
    correction = torch.zeros(len(graph['edges']), dtype=torch.float64, requires_grad=True)
    latency = torch.tensor([s['latency_prior'] for s in graph['sources']], dtype=torch.float64, requires_grad=True)
    log_base = torch.full((n,), -5., dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([log_type, correction, latency, log_base], lr=args.lr)
    history = []
    for epoch in range(args.epochs):
        loss = torch.zeros((), dtype=torch.float64)
        mean = (log_type[type_index] + correction).exp()
        scale = mean / shape
        for episode in episodes:
            if episode.get('split', 'train') != 'train':
                continue
            events = [e for e in episode['events'] if e['time'] <= episode['horizon']]
            times = torch.tensor([e['time'] for e in events], dtype=torch.float64)
            sensor = torch.tensor([e['source'] for e in events], dtype=torch.long)
            nodes = torch.tensor([e['node'] for e in events], dtype=torch.long)
            physical = times - latency[sensor]
            end = float(episode['horizon'])
            intensity = log_base.exp()[nodes]
            origin = float(episode.get('start', 0.))
            integral = log_base.exp().sum() * (end - origin)
            for j, edge in enumerate(graph['edges']):
                upstream = physical[nodes == edge['u']]
                downstream = physical[nodes == edge['v']]
                if len(upstream):
                    delta = downstream[:, None] - upstream[None]
                    positive = delta > 0
                    delta = delta.clamp_min(1e-8)
                    density = torch.exp((shape[j] - 1) * delta.log() - delta / scale[j]
                                        - torch.lgamma(shape[j]) - shape[j] * scale[j].log()) * positive
                    source_reliability = torch.tensor([graph['sources'][int(s)].get('reliability', 1.) for s in sensor[nodes == edge['u']]], dtype=torch.float64)
                    intensity = intensity.index_add(0, torch.nonzero(nodes == edge['v'], as_tuple=True)[0], (density * source_reliability).sum(1))
                    residual = (end - upstream).clamp_min(0) / scale[j]
                    initial = (origin - upstream).clamp_min(0) / scale[j]
                    integral = integral + ((torch.special.gammainc(shape[j], residual) - torch.special.gammainc(shape[j], initial)) * source_reliability).sum()
            loss = loss + integral - intensity.clamp_min(1e-12).log().sum()
        priors = torch.tensor([s['latency_prior'] for s in graph['sources']], dtype=torch.float64)
        widths = torch.tensor([max(s.get('latency_sd', 1.), 1e-6) for s in graph['sources']], dtype=torch.float64)
        loss = loss + .5 * ((latency - priors) / widths).square().sum() + args.local_penalty * correction.square().sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            for j, source in enumerate(graph['sources']):
                low, high = source['latency_bounds']
                latency[j].clamp_(low, high)
            latency[0] = graph['sources'][0]['latency_prior']
        history.append(float(loss.detach()))
    state = {'graph': graph, 'edge_mean': (log_type[type_index] + correction).exp().detach().tolist(),
             'latency': latency.detach().tolist(), 'baseline': log_base.exp().detach().tolist(), 'history': history}
    Path(args.output).write_text(json.dumps(state, indent=2))


def gamma_log_density(delta, mean, variance):
    if delta <= 0:
        return -100.
    shape, scale = mean * mean / variance, variance / mean
    return (shape - 1) * np.log(delta) - delta / scale - gammaln(shape) - shape * np.log(scale)


def decode(episode, state, variant, depth, missing_penalty):
    graph = state['graph']
    events = [e for e in episode['events'] if e['time'] <= episode['horizon']]
    if not events:
        raise ValueError('A diagnosis horizon must contain at least one event')
    times = np.array([e['time'] - (0 if variant == 'raw' else state['latency'][e['source']]) for e in events])
    nodes = {e['node'] for e in events}
    rankings = []
    for root in range(len(graph['nodes'])):
        candidates = {node: list(paths(graph, root, node, depth)) for node in nodes}
        if any(not candidate for candidate in candidates.values()):
            continue
        early = min(times)
        root_event = [times[i] for i, e in enumerate(events) if e['node'] == root]
        root_times = [min(root_event)] if root_event else np.linspace(early - argsafe_delay(state, depth), early - 1e-3, 31)
        for root_time in root_times:
            score, chosen, unsupported = 0., set(), set()
            for i, event in enumerate(events):
                best, best_path, best_missing = -np.inf, [], set()
                for candidate in candidates[event['node']]:
                    means = np.array([state['edge_mean'][j] for j in candidate])
                    variance = sum(state['edge_mean'][j] ** 2 / graph['kernels'][graph['edges'][j]['type']]['shape'] for j in candidate)
                    hidden = {graph['edges'][j]['u'] for j in candidate if graph['edges'][j]['u'] not in nodes}
                    if variant == 'no_missing' and hidden:
                        continue
                    if candidate:
                        value = gamma_log_density(times[i] - root_time, means.sum(), max(variance, 1e-6))
                    else:
                        value = -abs(times[i] - root_time)
                    reliability = 1. if variant == 'no_reliability' else graph['sources'][event['source']].get('reliability', 1.)
                    value = reliability * value - .1 * len(candidate) - missing_penalty * len(hidden)
                    if value > best:
                        best, best_path, best_missing = value, candidate, hidden
                score += best
                chosen.update(best_path)
                unsupported.update(best_missing)
            if np.isfinite(score):
                rankings.append({'root': root, 'score': float(score), 'root_time': float(root_time),
                                 'path': sorted(chosen), 'unsupported_nodes': sorted(unsupported)})
    best_by_root = {}
    for candidate in rankings:
        root = candidate['root']
        if root not in best_by_root or candidate['score'] > best_by_root[root]['score']:
            best_by_root[root] = candidate
    ranked = sorted(best_by_root.values(), key=lambda v: -v['score'])
    if not ranked:
        raise ValueError('No admissible rooted explanation; increase depth or inspect topology')
    return ranked, times.tolist()


def argsafe_delay(state, depth):
    return max(state['edge_mean']) * depth * 3


def event_nll(episode, state, physical, edges):
    graph = state['graph']
    events = [e for e in episode['events'] if e['time'] <= episode['horizon']]
    from scipy.special import gammainc
    origin = episode.get('start', 0.)
    integral = sum(state['baseline']) * (episode['horizon'] - origin)
    loglike = 0.
    for i, event in enumerate(events):
        intensity = state['baseline'][event['node']]
        for j in edges:
            edge = graph['edges'][j]
            if edge['v'] != event['node']:
                continue
            mean = state['edge_mean'][j]
            shape = graph['kernels'][edge['type']]['shape']
            for k, upstream in enumerate(events):
                if upstream['node'] == edge['u'] and physical[k] < physical[i]:
                    reliability = graph['sources'][upstream['source']].get('reliability', 1.)
                    intensity += reliability * np.exp(gamma_log_density(physical[i] - physical[k], mean, mean * mean / shape))
        loglike += np.log(max(intensity, 1e-12))
    for j in edges:
        edge = graph['edges'][j]
        shape = graph['kernels'][edge['type']]['shape']
        for k, upstream in enumerate(events):
            if upstream['node'] == edge['u']:
                reliability = graph['sources'][upstream['source']].get('reliability', 1.)
                integral += reliability * (gammainc(shape, max(0, episode['horizon'] - physical[k]) * shape / state['edge_mean'][j]) - gammainc(shape, max(0, origin - physical[k]) * shape / state['edge_mean'][j]))
    return float((integral - loglike) / len(events))


def evaluate(args):
    state = json.loads(Path(args.model).read_text())
    episodes = json.loads(Path(args.data).read_text())
    rng = np.random.default_rng(args.seed)
    records = []
    for source in episodes:
        if source.get('split', 'test') != 'test':
            continue
        episode = json.loads(json.dumps(source))
        removed_sources = set(rng.choice(len(state['graph']['sources']), round(args.missing * len(state['graph']['sources'])), replace=False).tolist())
        episode['events'] = [e for e in episode['events'] if e['source'] not in removed_sources]
        for event in episode['events']:
            event['time'] += args.clock_offset + args.latency_drift * event['source']
        episode['start'] = episode.get('start', 0.) + args.clock_offset
        episode['horizon'] += args.clock_offset
        begin = time.perf_counter()
        ranked, physical = decode(episode, state, args.variant, args.depth, args.missing_penalty)
        record = {'id': episode['id'], 'ranking': ranked, 'physical_time': physical,
                  'latency_ms': (time.perf_counter() - begin) * 1000, 'nll': event_nll(episode, state, physical, ranked[0]['path']),
                  'unsupported_mass': len(ranked[0]['unsupported_nodes']), 'decision_time': episode['horizon']}
        if 'root' in episode:
            root_rank = next((j + 1 for j, v in enumerate(ranked) if v['root'] == episode['root']), float('inf'))
            record.update(hit1=root_rank == 1, hit3=root_rank <= 3, reciprocal_rank=1 / root_rank)
            record['time_to_diagnosis'] = episode['horizon'] - args.clock_offset - episode.get('injection_time', 0.)
        if 'path' in episode:
            truth, prediction = set(episode['path']), set(ranked[0]['path'])
            overlap = len(truth & prediction)
            precision, recall = overlap / max(1, len(prediction)), overlap / max(1, len(truth))
            record.update(path_precision=precision, path_recall=recall, path_f1=2 * precision * recall / max(1e-12, precision + recall))
        if 'edge_delay' in episode:
            record['lag_mae'] = float(np.mean([abs(state['edge_mean'][int(k)] - v) for k, v in episode['edge_delay'].items()]))
        records.append(record)
    keys = ['hit1', 'hit3', 'reciprocal_rank', 'path_precision', 'path_recall', 'path_f1', 'lag_mae', 'nll', 'unsupported_mass', 'time_to_diagnosis', 'latency_ms']
    result = {k: float(np.mean([r[k] for r in records if k in r])) for k in keys if any(k in r for r in records)}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'episodes.json').write_text(json.dumps(records, indent=2))
    (out / 'metrics.json').write_text(json.dumps(result, indent=2))
    plt.bar(np.arange(len(records)), [r['nll'] for r in records])
    plt.xlabel('Fault episode')
    plt.ylabel('Event NLL per observed event')
    plt.tight_layout()
    plt.savefig(out / 'likelihood.png', dpi=180)
    plt.close()


def extract(args):
    data = np.load(args.source)
    training = data['split'] == 'train'
    if 'normal' in data:
        training &= data['normal'].astype(bool)
    mean = data['x'][training].mean(0)
    std = data['x'][training].std(0).clip(1e-6)
    z = np.abs((data['x'] - mean) / std)
    episodes = []
    for identifier in np.unique(data['episode']):
        ids = np.flatnonzero(data['episode'] == identifier)
        if len(np.unique(data['split'][ids])) != 1:
            raise ValueError('An episode cannot cross splits')
        previous = np.zeros(z.shape[1], dtype=bool)
        events = []
        for i in ids:
            active = z[i] >= args.threshold
            for node in np.flatnonzero(active & ~previous):
                events.append({'node': int(node), 'source': int(node), 'time': float(data['time'][i])})
            previous = active
        episodes.append({'id': str(identifier), 'split': str(data['split'][ids[0]]),
                         'horizon': float(data['time'][ids[-1]]), 'events': events})
    Path(args.output).write_text(json.dumps(episodes, indent=2))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('extract')
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--threshold', type=float, default=3.)
    p = sub.add_parser('fit')
    for name in ['data', 'graph', 'output']:
        p.add_argument('--' + name, required=True)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--lr', type=float, default=.01)
    p.add_argument('--local-penalty', type=float, default=1.)
    p = sub.add_parser('evaluate')
    for name in ['data', 'model', 'output']:
        p.add_argument('--' + name, required=True)
    p.add_argument('--variant', choices=['full', 'raw', 'no_missing', 'no_reliability'], default='full')
    p.add_argument('--depth', type=int, default=6)
    for name, default in [('missing', 0.), ('missing-penalty', 1.), ('clock-offset', 0.), ('latency-drift', 0.)]:
        p.add_argument('--' + name, type=float, default=default)
    p.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    {'extract': extract, 'fit': fit, 'evaluate': evaluate}[args.command](args)
