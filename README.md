# 🧠 LLM Inference Interface

A lightweight Dockerized interface for running a light weight LLM (**Llama-3.2-3B-Instruct with 4b quantization**) inference using [Gradio](https://www.gradio.app/). This interface makes it easy to deploy and interact with your model via a clean web UI.

---

## 🚀 Getting Started

### 📄 Optional: Configure `.env`

You can create a `.env` file to customize the server port and host:

```env
GRADIO_SERVER_PORT=8080
GRADIO_SERVER_NAME=0.0.0.0
```

---

## 🛠️ Build the Docker Image
To build the Docker container from the Dockerfile, run the following command:
`docker build -t gradio-app .`

---

## 🐳 Run Docker Container
To run the Docker container you just built, use this command:
`docker run --gpus all -p 8080:8080 -it gradio-app`

---

📌 **Note**:
- Once running, access the chatbot in your browser at: http://localhost:8080

- You can change the default port or host by editing the .env file or modifying the Dockerfile.

- To persist chat history between sessions, mount a local volume:
`docker run -v <local_path>:/src --gpus all -p 8080:8080 -it gradio-app`
Make sure history.json exists at /src/history.json

---

💬 ## Chat Interface
Here’s a preview of the chatbot interface when the container is running:
![Chat Interface](Interface.png)