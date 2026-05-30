# DAVE — Design Agent for Visual Exploration

> A live, physical-to-digital design tool that bridges 3D printed massing models with AI-driven parametric generation in Rhino/Grasshopper.

![DAVE demo](./assets/demo.gif)
*Place physical blocks, watch the model update in real time.*

---

## What is DAVE?

DAVE lets architects and urban designers **physically manipulate 3D printed massing blocks** on a table, while a camera tracks the layout and streams it into Rhino as live 2D outlines. From there, an AI agent (Claude) can generate parks, pedestrian paths, bridges, visibility analyses, and more — all in real time, just by talking to it.

Built at AECTech

---

## Demo

![CV edge detection pipeline](./assets/edge_detection.gif)
*OpenCV edge detection converting physical blocks to 2D outlines.*

![Rhino live update](./assets/rhino_live.gif)
*Outlines updating in Rhino as blocks are moved.*

![Claude generating geometry](./assets/claude_agent.gif)
*Asking DAVE to generate a park between two blocks.*

---

## Architecture

![System architecture diagram](./assets/architecture.png)

| Layer | What it does |
|---|---|
| Physical | 3D printed massing blocks on a calibration mat |
| Computer Vision | OpenCV edge detection → polygon extraction → socket stream |
| Rhino / Grasshopper | Receives live outlines, runs parametric GH scripts |
| AI Agent | Claude via MCP + Swiftlet, calls GH tools from natural language |

---

## Getting started

### Prerequisites

- Python 3.11+
- Rhino 8 with Grasshopper
- [Swiftlet](https://github.com/your-swiftlet-link) MCP plugin for Rhino
- Claude/LLM 
- A webcam 

### Installation

```bash
git clone https://github.com/your-org/dave
cd dave
pip install -r requirements.txt
```

### Running the CV pipeline

### Connecting to Rhino

### Starting the AI agent

## CV calibration tips

## Grasshopper tools

DAVE exposes the following tools to the Claude agent via Swiftlet MCP:

| Tool | Description |
|---|---|
| `generate_park` | Creates green space geometry between specified blocks |
| `draw_pedestrian_path` | Calculates and draws walking routes across the site |
| `generate_bridge` | Connects two massing blocks with a bridge element |

## Repo structure

---

## Team

---

## License

MIT
