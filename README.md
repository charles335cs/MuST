# MuST: Robust Image Watermarking for Multi-Source Tracing — Unofficial Implementation

This repository provides an **unofficial implementation and deployment** of [MuST](https://dl.acm.org/doi/10.1609/aaai.v38i6.28344), a robust localized image watermarking framework.  

> ⚠️ Disclaimer: This is **not an official release**.
The official code does not provide pretrained models or testing scripts and contains many bugs. To address these issues, we redeployed MuST, fixed the bugs, and provide a model trained for 300 epochs on the SOIM dataset.
---

## 🔗 Pretrained Models
- [MuST-SOIM (300 epochs)](https://your-link-here.com)

---

## 🚀 Training
~~~bash
# Train MuST on SOIM dataset, you can start from scratch or load pretrained models.
python main.py
~~~

## 🚀 Testing
We split the encoding and decoding process,
~~~bash
# For watermark encoding:
python encode.py
~~~

~~~bash
# For watermark decoding:
python decode.py
~~~
