## Current Progress

### Ensemble Networks
- [x] All policy network is shared.

### CHATRPO
- [x] Actor class implemented
- [x] Agents have individual buffers
- [x] At 1 update step, actor accesses buffer for all agents in the class sequentially.

## Roadmap

### Ensemble Networks
- [ ] Observation vector changes: Embedding and MLP should not be shared.
- [ ] Apply few-shot learning to those networks.

### CHATRPO
- [ ] Implement Class to critics via value decomposition.
- [ ] Have the choice of not using a central critic and only class critic.

### Research LAB
- [ ] Class based critic experiments.
- [ ] Env wrapper to create sensing radius.
- [ ] Graph Information Aggregation of local information.

## Code Examples

### Ensemble Networks
```bash
python examples/train.py --algo embd --env pettingzoo_mpe --exp_name embd
```

### CHATRPO
```bash
python examples/train.py --algo chatrpo --env pettingzoo_mpe
```
