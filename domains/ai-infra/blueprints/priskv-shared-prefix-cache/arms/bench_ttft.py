#!/usr/bin/env python3
"""Shared-prefix TTFT benchmark for PrisKV A/B/C experiment.
Generates N sessions; a fraction `reuse` share one large system prompt (cache-hittable),
the rest get unique prefixes. Measures streaming TTFT via /v1/completions stream=true.
Sets X-Prefix-Key header = hash of the shared-prefix region (for Arm B prefix routing).
"""
import sys, json, time, hashlib, urllib.request, random, statistics, argparse

def make_prompt(shared_prefix, uniq):
    return f"{shared_prefix}\n\nUser question {uniq}: explain this in one sentence."

def stream_ttft(base, prompt, prefix_key, max_tokens=8):
    body = json.dumps({"model":"Qwen3-32B-FP8","prompt":prompt,"max_tokens":max_tokens,
                       "temperature":0,"stream":True}).encode()
    req = urllib.request.Request(base+"/v1/completions", data=body,
            headers={"Content-Type":"application/json","X-Prefix-Key":prefix_key})
    t0=time.time(); ttft=None; ntok=0; txt=""
    with urllib.request.urlopen(req, timeout=300) as r:
        for line in r:
            line=line.decode().strip()
            if not line or not line.startswith("data:"): continue
            d=line[5:].strip()
            if d=="[DONE]": break
            try: obj=json.loads(d)
            except: continue
            tok=obj.get("choices",[{}])[0].get("text","")
            if tok:
                if ttft is None: ttft=time.time()-t0
                ntok+=1; txt+=tok
    return ttft, time.time()-t0, txt

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base",default="http://localhost:8080")
    ap.add_argument("--arm",required=True)
    ap.add_argument("--reuse",type=float,default=0.7)
    ap.add_argument("--sessions",type=int,default=40)
    ap.add_argument("--prefix-tokens",type=int,default=800)  # >128 floor
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    random.seed(10)
    # one shared prefix (big) + pool of unique prefixes
    shared=" ".join(["The system operates under the following policy directive."]*(a.prefix_tokens//8))
    uniq_prefixes={}
    def uniq_prefix(i):
        if i not in uniq_prefixes:
            uniq_prefixes[i]=" ".join([f"Unique-context-{i} token{random.randint(0,9999)}."]*(a.prefix_tokens//6))
        return uniq_prefixes[i]
    results=[]
    # warmup: prime the shared prefix once per replica (2) so steady-state reflects warm cache
    for w in range(4):
        stream_ttft(a.base, make_prompt(shared,f"warm{w}"), hashlib.md5(shared.encode()).hexdigest())
    for i in range(a.sessions):
        if random.random()<a.reuse:
            pfx=shared; key=hashlib.md5(shared.encode()).hexdigest(); kind="shared"
        else:
            u=i%7; pfx=uniq_prefix(u); key=hashlib.md5(pfx.encode()).hexdigest(); kind="unique"
        ttft,e2e,txt=stream_ttft(a.base, make_prompt(pfx,i), key)
        results.append({"i":i,"kind":kind,"ttft_ms":round((ttft or 0)*1000,1),"e2e_ms":round(e2e*1000,1)})
    ttfts=[r["ttft_ms"] for r in results if r["ttft_ms"]>0]
    shared_ttfts=[r["ttft_ms"] for r in results if r["kind"]=="shared" and r["ttft_ms"]>0]
    summary={"arm":a.arm,"reuse":a.reuse,"sessions":a.sessions,"prefix_tokens":a.prefix_tokens,
        "ttft_p50":round(statistics.median(ttfts),1),
        "ttft_p99":round(sorted(ttfts)[int(len(ttfts)*0.99)-1],1) if len(ttfts)>1 else ttfts[0],
        "ttft_shared_p50":round(statistics.median(shared_ttfts),1) if shared_ttfts else None,
        "n_shared":len(shared_ttfts),"n_total":len(ttfts)}
    json.dump({"summary":summary,"results":results}, open(a.out,"w"), indent=2)
    print(json.dumps(summary))

if __name__=="__main__": main()
