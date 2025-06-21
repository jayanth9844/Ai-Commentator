# ♟️ AI Commentator for Chess

<p align="center">
  <img src="https://img.shields.io/badge/Project-Type:AI_Chess_Commentary-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/Status-Completed-brightgreen?style=flat-square" />
  <img src="https://img.shields.io/badge/Made%20at-REVA%20University-orange?style=flat-square" />
</p>

---

## 🎯 Project Overview

**AI Commentator** is an intelligent, offline-compatible chess analysis system that generates real-time, human-like commentary for chess moves. It integrates the Stockfish 16 chess engine, a feature extraction pipeline, and a fine-tuned GPT-2 transformer model to provide insightful and educational move analysis. Commentary is delivered both as on-screen text and via speech using `pyttsx3`.

---

## 📽️ Demo Video

👉 [Click here to watch the demo](#)  
_(Replace `#` with your YouTube or Drive link)_

---

## 🖼️ Key Visuals

<table>
<tr>
<td align="center"><strong>System Architecture</strong><br><img src="images/System Architecture.png" width="500"/></td>
<td align="center"><strong>Model Loss Curve</strong><br><img src="images/Model Loss Curve.jpg" width="500"/></td>
</tr>
<tr>
<td align="center"><strong>GUI Interface with Commentary</strong><br><img src="images/GUI Interface with Commentary.png" width="500"/></td>
</tr>
</table>

> ℹ️ Replace the image paths with your actual file URLs.

---

## 🚀 Features

- 🧠 Real-time move evaluation using **Stockfish 16**
- 🗣️ Natural language commentary using **fine-tuned GPT-2**
- 🔊 Offline **Text-to-Speech** via `pyttsx3`
- 📦 Fully **modular & offline-compatible** (no internet needed)
- 🎮 Interactive GUI built with **Pygame**
- ♿ Accessibility support for **visually impaired users**

---

## 🔧 Tech Stack

| Component                  | Tool/Library                        |
|---------------------------|-------------------------------------|
| Chess Engine              | [Stockfish 16](https://stockfishchess.org) |
| GUI & Game Logic          | `pygame`, `python-chess`            |
| NLP Model                 | [GPT-2](https://openaipublic.blob.core.windows.net/gpt-2/models) |
| Feature Extraction        | Custom pipeline on FEN strings      |
| TTS (Speech)              | `pyttsx3` (offline)                 |
| Data Source               | [Lichess.org](https://lichess.org/) |

---

## 📐 System Architecture

```mermaid
graph TD;
    A[User Move on GUI] --> B[python-chess]
    B --> C[Stockfish 16 Evaluation]
    C --> D[Feature Extraction + Formatted String]
    D --> E[Fine-tuned GPT-2]
    E --> F1[On-screen Commentary]
    E --> F2[pyttsx3 Speech Output]
