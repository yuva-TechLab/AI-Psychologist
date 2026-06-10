"""
Module 4: AI Prediction Model (NumPy implementation)
Three models: Markov Chain, LSTM, Transformer
"""
import json, numpy as np, logging
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
SEQ_DIR  = BASE_DIR / "data" / "sequences"
MDL_DIR  = BASE_DIR / "models"
MDL_DIR.mkdir(parents=True, exist_ok=True)

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)
def relu(x): return np.maximum(0, x)
def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

class MarkovChainModel:
    def __init__(self, order=1):
        self.order = order
        self.transitions = defaultdict(lambda: defaultdict(float))
        self.vocab = {}; self.inv_vocab = {}
    def fit(self, sessions, vocab):
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        for session in sessions:
            for i in range(len(session) - self.order):
                ctx = tuple(session[i:i+self.order])
                tgt = session[i+self.order]
                self.transitions[ctx][tgt] += 1
        for ctx in self.transitions:
            total = sum(self.transitions[ctx].values())
            for t in self.transitions[ctx]: self.transitions[ctx][t] /= total
        logger.info(f"Markov fitted: {len(self.transitions)} contexts")
    def predict(self, token_seq, top_k=3):
        stages = [self.inv_vocab.get(int(t), "<UNK>") for t in token_seq]
        ctx    = tuple(stages[-self.order:])
        nexts  = dict(self.transitions.get(ctx, {}))
        if not nexts:
            fb    = [s for s in self.vocab if s not in ("<PAD>","<UNK>","Benign")]
            nexts = {s: 1.0/len(fb) for s in fb}
        sorted_n = sorted(nexts.items(), key=lambda x: -x[1])
        best, conf = sorted_n[0]
        topk = [(self.vocab.get(s, 1), p) for s, p in sorted_n[:top_k]]
        return self.vocab.get(best, 1), conf, topk
    def evaluate(self, sessions):
        correct = total = 0
        for session in sessions:
            for i in range(len(session) - self.order):
                ctx = [self.vocab.get(s, 1) for s in session[i:i+self.order]]
                true_tok = self.vocab.get(session[i+self.order], 1)
                pred_tok, _, _ = self.predict(ctx)
                if pred_tok == true_tok: correct += 1
                total += 1
        acc = correct / total if total > 0 else 0
        return {"accuracy": acc, "correct": correct, "total": total}

class LSTMCell:
    def __init__(self, in_sz, hid_sz, seed=0):
        rng = np.random.default_rng(seed)
        sc  = np.sqrt(2.0/(in_sz+hid_sz))
        self.W = rng.normal(0, sc, (hid_sz*4, in_sz+hid_sz))
        self.b = np.zeros(hid_sz*4)
        self.H = hid_sz
    def forward(self, x, h, c):
        g = self.W @ np.concatenate([x, h]) + self.b
        H = self.H
        i = sigmoid(g[0*H:1*H]); f = sigmoid(g[1*H:2*H])
        gg= np.tanh(g[2*H:3*H]); o = sigmoid(g[3*H:4*H])
        c2 = f*c + i*gg; h2 = o*np.tanh(c2)
        return h2, c2

class LSTMPredictor:
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64, num_classes=None, seed=42):
        rng = np.random.default_rng(seed)
        nc  = num_classes or vocab_size
        self.vocab_size = vocab_size; self.hidden_dim = hidden_dim; self.nc = nc
        self.embed = rng.normal(0, 0.1, (vocab_size, embed_dim))
        self.lstm1 = LSTMCell(embed_dim, hidden_dim, seed)
        self.lstm2 = LSTMCell(hidden_dim, hidden_dim, seed+1)
        s1 = np.sqrt(2.0/hidden_dim)
        self.W1 = rng.normal(0, s1, (hidden_dim, hidden_dim)); self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(0, s1, (nc, hidden_dim));         self.b2 = np.zeros(nc)
        self.inv_vocab = {}
    def _fwd(self, seq):
        h1=c1=np.zeros(self.hidden_dim); h2=c2=np.zeros(self.hidden_dim)
        for t in seq:
            x = self.embed[min(int(t), self.vocab_size-1)]
            h1,c1 = self.lstm1.forward(x,h1,c1)
            h2,c2 = self.lstm2.forward(h1,h2,c2)
        z = relu(self.W1@h2+self.b1)
        return self.W2@z+self.b2, h2
    def predict_proba(self, seq): return softmax(self._fwd(seq)[0])
    def predict(self, seq, top_k=3):
        probs = self.predict_proba(seq)
        top   = np.argsort(probs)[::-1][:top_k]
        return int(top[0]), float(probs[top[0]]), [(int(i), float(probs[i])) for i in top]
    def train_epoch(self, X, y, lr=0.05):
        total = 0.0
        for idx in np.random.permutation(len(X)):
            seq = X[idx]; tgt = int(y[idx])
            logits, h2 = self._fwd(seq)
            probs = softmax(logits); total -= np.log(probs[tgt]+1e-9)
            dl = probs.copy(); dl[tgt] -= 1.0
            dW2=np.outer(dl,relu(self.W1@h2+self.b1)); db2=dl
            dz=(self.W2.T@dl)*(relu(self.W1@h2+self.b1)>0)
            dW1=np.outer(dz,h2); db1=dz
            for g in [dW2,db2,dW1,db1]: np.clip(g,-1,1,out=g)
            self.W2-=lr*dW2; self.b2-=lr*db2; self.W1-=lr*dW1; self.b1-=lr*db1
        return total/len(X)

class MultiHeadAttn:
    def __init__(self, D, H, seed=0):
        assert D%H==0
        rng=np.random.default_rng(seed); sc=np.sqrt(2.0/D)
        self.H=H; self.dk=D//H
        self.Wq=rng.normal(0,sc,(D,D)); self.Wk=rng.normal(0,sc,(D,D))
        self.Wv=rng.normal(0,sc,(D,D)); self.Wo=rng.normal(0,sc,(D,D))
    def forward(self, x):
        T,D=x.shape; H,dk=self.H,self.dk
        Q=(x@self.Wq.T).reshape(T,H,dk).transpose(1,0,2)
        K=(x@self.Wk.T).reshape(T,H,dk).transpose(1,0,2)
        V=(x@self.Wv.T).reshape(T,H,dk).transpose(1,0,2)
        sc=Q@K.transpose(0,2,1)/np.sqrt(dk)
        at=softmax(sc); ctx=(at@V).transpose(1,0,2).reshape(T,D)
        return ctx@self.Wo.T

class TFBlock:
    def __init__(self, D, H, ff, seed=0):
        rng=np.random.default_rng(seed)
        self.attn=MultiHeadAttn(D,H,seed)
        self.W1=rng.normal(0,np.sqrt(2/D),(ff,D)); self.b1=np.zeros(ff)
        self.W2=rng.normal(0,np.sqrt(2/ff),(D,ff)); self.b2=np.zeros(D)
    def ln(self, x, eps=1e-5):
        m=x.mean(-1,keepdims=True); s=x.std(-1,keepdims=True)+eps
        return (x-m)/s
    def forward(self, x):
        x=self.ln(x+self.attn.forward(x))
        return self.ln(x+(relu(x@self.W1.T+self.b1)@self.W2.T+self.b2))

class TransformerPredictor:
    def __init__(self, vocab_size, embed_dim=32, nhead=4, num_layers=2,
                 dim_ff=64, num_classes=None, max_seq=32, seed=42):
        rng=np.random.default_rng(seed)
        nc=num_classes or vocab_size
        D=nhead*max(1,embed_dim//nhead)
        self.vocab_size=vocab_size; self.D=D; self.nc=nc
        self.embed=rng.normal(0,0.1,(vocab_size,D))
        self.pos  =rng.normal(0,0.1,(max_seq,D))
        self.blocks=[TFBlock(D,nhead,dim_ff,seed+i) for i in range(num_layers)]
        s=np.sqrt(2/D)
        self.W1=rng.normal(0,s,(D,D)); self.b1=np.zeros(D)
        self.W2=rng.normal(0,s,(nc,D)); self.b2=np.zeros(nc)
        self.inv_vocab={}
    def _fwd(self, seq):
        T=len(seq); idx=[min(int(t),self.vocab_size-1) for t in seq]
        x=self.embed[idx]+self.pos[:T]
        for bl in self.blocks: x=bl.forward(x)
        p=x.mean(0); z=relu(p@self.W1.T+self.b1)
        return self.W2@z+self.b2, p
    def predict_proba(self, seq): return softmax(self._fwd(seq)[0])
    def predict(self, seq, top_k=3):
        probs=self.predict_proba(seq); top=np.argsort(probs)[::-1][:top_k]
        return int(top[0]), float(probs[top[0]]), [(int(i),float(probs[i])) for i in top]
    def train_epoch(self, X, y, lr=0.05):
        total=0.0
        for idx in np.random.permutation(len(X)):
            seq=X[idx]; tgt=int(y[idx])
            logits,p=self._fwd(seq); probs=softmax(logits); total-=np.log(probs[tgt]+1e-9)
            dl=probs.copy(); dl[tgt]-=1.0
            z=relu(p@self.W1.T+self.b1)
            dW2=np.outer(dl,z); db2=dl
            dz=(self.W2.T@dl)*(z>0)
            dW1=np.outer(dz,p); db1=dz
            for g in [dW2,db2,dW1,db1]: np.clip(g,-1,1,out=g)
            self.W2-=lr*dW2; self.b2-=lr*db2; self.W1-=lr*dW1; self.b1-=lr*db1
        return total/len(X)

def train_model(model, X, y, epochs=40, lr=0.05, val_split=0.2, name="model"):
    split=int(len(X)*(1-val_split)); idx=np.random.permutation(len(X))
    Xtr,ytr=X[idx[:split]],y[idx[:split]]; Xv,yv=X[idx[split:]],y[idx[split:]]
    history={"train_loss":[],"val_acc":[]}; best_acc=0.0
    logger.info(f"Training {name}: {len(Xtr)} train | {len(Xv)} val | {epochs} epochs")
    for ep in range(1,epochs+1):
        loss=model.train_epoch(Xtr,ytr,lr=lr)
        correct=sum(1 for s,t in zip(Xv,yv) if model.predict(s)[0]==int(t))
        acc=correct/len(yv); history["train_loss"].append(round(float(loss),4)); history["val_acc"].append(round(acc,4))
        if acc>best_acc: best_acc=acc
        if ep%10==0 or ep==1: logger.info(f"  Epoch {ep:3d}/{epochs}  loss={loss:.4f}  val_acc={acc:.4f}")
    logger.info(f"{name} best val_acc={best_acc:.4f}"); return history

def evaluate_model(model, X, y, inv_vocab):
    preds=[model.predict(s)[0] for s in X]
    acc=sum(p==int(t) for p,t in zip(preds,y))/len(y)
    return {"accuracy": acc, "predictions": preds}

def save_numpy(model, name, meta):
    path=MDL_DIR/f"{name}.npz"
    if isinstance(model, LSTMPredictor):
        np.savez(path,embed=model.embed,W1=model.W1,b1=model.b1,W2=model.W2,b2=model.b2)
    elif isinstance(model, TransformerPredictor):
        np.savez(path,embed=model.embed,pos=model.pos,W1=model.W1,b1=model.b1,W2=model.W2,b2=model.b2)
    with open(MDL_DIR/f"{name}_meta.json","w") as f: json.dump(meta,f,indent=2)
    logger.info(f"Saved {name} -> {path}")

if __name__=="__main__":
    import sys; sys.path.insert(0,str(BASE_DIR))
    with open(SEQ_DIR/"vocabulary.json") as f: vdata=json.load(f)
    with open(SEQ_DIR/"attack_sessions.json") as f: sessions=json.load(f)
    vocab={k:int(v) for k,v in vdata["vocab"].items()}
    inv_vocab={int(k):v for k,v in vdata["inv_vocab"].items()}
    X=np.load(SEQ_DIR/"X_sequences.npy"); y=np.load(SEQ_DIR/"y_targets.npy")
    vs=len(vocab)
    test_seq=[vocab.get("Reconnaissance",2),vocab.get("Credential Access",3),vocab.get("Exploitation",4)]

    print(f"\n{'='*55}\n  MODULE 4 — Model Training\n{'='*55}")
    print(f"  Vocab={vs}  Pairs={len(X)}\n")
    results={}

    print("[ 1/3 ] Markov Chain ...")
    markov=MarkovChainModel(order=1); markov.fit(sessions,vocab)
    r=markov.evaluate(sessions); results["Markov Chain"]=r["accuracy"]
    p,c,_=markov.predict(test_seq[:2])
    print(f"  Accuracy: {r['accuracy']:.4f}  |  Pred: Recon->Cred -> {inv_vocab.get(p,'?')} ({c:.3f})\n")

    print("[ 2/3 ] LSTM ...")
    lstm=LSTMPredictor(vocab_size=vs,embed_dim=32,hidden_dim=64,num_classes=vs)
    h1=train_model(lstm,X,y,epochs=40,lr=0.05,name="LSTM")
    r2=evaluate_model(lstm,X,y,inv_vocab); results["LSTM"]=r2["accuracy"]
    save_numpy(lstm,"lstm_attack_predictor",{"vocab":vocab,"inv_vocab":{str(k):v for k,v in inv_vocab.items()},"model":"lstm","vocab_size":vs})
    p,c,_=lstm.predict(test_seq)
    print(f"  Accuracy: {r2['accuracy']:.4f}  |  Pred: Recon->Cred->Exploit -> {inv_vocab.get(p,'?')} ({c:.3f})\n")

    print("[ 3/3 ] Transformer ...")
    tfmr=TransformerPredictor(vocab_size=vs,embed_dim=32,nhead=4,num_layers=2,dim_ff=64,num_classes=vs)
    h2=train_model(tfmr,X,y,epochs=40,lr=0.05,name="Transformer")
    r3=evaluate_model(tfmr,X,y,inv_vocab); results["Transformer"]=r3["accuracy"]
    save_numpy(tfmr,"transformer_attack_predictor",{"vocab":vocab,"inv_vocab":{str(k):v for k,v in inv_vocab.items()},"model":"transformer","vocab_size":vs})
    p,c,_=tfmr.predict(test_seq)
    print(f"  Accuracy: {r3['accuracy']:.4f}  |  Pred: Recon->Cred->Exploit -> {inv_vocab.get(p,'?')} ({c:.3f})\n")

    print(f"{'='*45}\n  MODEL COMPARISON\n{'='*45}")
    print(f"  {'Model':<22} {'Accuracy':>10}  Bar")
    print(f"  {'-'*42}")
    for nm,acc in results.items():
        bar="█"*int(acc*25)
        print(f"  {nm:<22} {acc:>10.4f}  {bar}")
    print(f"{'='*45}")

    with open(MDL_DIR/"training_history.json","w") as f:
        json.dump({"lstm":h1,"transformer":h2},f,indent=2)

    print(f"\nModule 4 complete. Models saved to: {MDL_DIR}")
    print(f"Next: Module 5 — Threat Scoring Engine\n")
