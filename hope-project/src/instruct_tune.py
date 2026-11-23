import os, time, math, random, torch
import torch.nn.functional as F
from datasets import load_dataset
from configs import get_config
from model import HOPE
from build_dataset import TokenizerWrapper, _generate_logic_puzzles

def build_instruct_samples(n_squad=6000, n_logic=6000):
    print('Streaming grounded reading-comprehension Q&A (rajpurkar/squad)...')
    ds = load_dataset('rajpurkar/squad', split='train', streaming=True)
    samples = []
    for row in ds:
        ctx = row['context'].strip()
        q = row['question'].strip()
        ans = row['answers']['text'][0].strip() if row['answers']['text'] else ''
        if not ans or len(ctx) > 1200:
            continue
        samples.append(f'[Reference 1]: {ctx}\n\nUser: {q}\nAssistant: {ans}')
        if len(samples) >= n_squad:
            break
    print(f'Generating {n_logic} pure logic & variable-tracking Q&A pairs...')
    for p in _generate_logic_puzzles(n_logic, seed=99):
        if 'Question:' in p and 'Answer:' in p:
            premise, rest = p.split('Question:', 1)
            q, ans = rest.split('Answer:', 1)
            samples.append(f'[Reference 1]: {premise.strip()}\n\nUser: {q.strip()}\nAssistant: {ans.strip()}')
    random.Random(42).shuffle(samples)
    return samples

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = get_config('local_200m')
    tok = TokenizerWrapper.for_gpt2()
    model = HOPE(cfg).to(device)
    ckpt = torch.load('checkpoints/local_200m_best.pt', map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.enable_gradient_checkpointing()
    model.train()
    samples = build_instruct_samples(6000, 6000)
    seqs = []
    for s in samples:
        ids = tok.encode(s) + [tok.eos_id]
        ids = ids[: cfg.seq_len + 1]
        if len(ids) < cfg.seq_len + 1:
            ids += [tok.pad_id] * (cfg.seq_len + 1 - len(ids))
        seqs.append(torch.tensor(ids, dtype=torch.long))
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.05)
    max_steps, batch_size, start = 1200, 8, time.time()
    print(f'Polishing {len(seqs):,} logic + search-extraction Q&A sequences ({max_steps} steps)...')
    for step in range(1, max_steps + 1):
        batch = torch.stack(random.sample(seqs, batch_size)).to(device)
        lr = 5e-5 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / max_steps)))
        for pg in opt.param_groups:
            pg['lr'] = lr
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, _ = model(batch[:, :-1], pad_token_id=tok.pad_id)
            loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), batch[:, 1:].reshape(-1), ignore_index=tok.pad_id)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or step == 1:
            print(f'  Step {step:>4}/{max_steps} | Loss: {loss.item():.4f} | Elapsed: {int(time.time()-start)}s')
    out = 'checkpoints/local_200m_chat.pt'
    torch.save({'model_state': model.state_dict(), 'step': max_steps, 'best_val_loss': loss.item()}, out)
    print(f'Saved chat-polished model to {out}')

if __name__ == '__main__':
    main()
