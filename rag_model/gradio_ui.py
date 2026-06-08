import requests
import gradio as gr

API_URL = "http://localhost:8000/query"


def ask_ros2(question: str) -> str:
    if not question or not question.strip():
        return "Please enter a ROS2 question."

    try:
        response = requests.post(
            API_URL,
            json={"question": question, "top_k": 5},
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()

        answer = data.get("answer", "")
        contexts = data.get("retrieved_context", [])

        source_lines = []
        for idx, context in enumerate(contexts, start=1):
            source = context.get("source") or "unknown"
            url = context.get("url") or "no-url"
            score = context.get("score")
            source_lines.append(f"[{idx}] {source} | {url} | score={score}")

        return f"{answer}\n\nRetrieved Sources:\n" + "\n".join(source_lines)

    except Exception as exc:
        return f"Backend query failed: {exc}"


demo = gr.Interface(
    fn=ask_ros2,
    inputs=gr.Textbox(
        label="Ask a ROS2 question",
        lines=4,
        placeholder="Example: How do I create a ROS2 publisher and subscriber in Python?",
    ),
    outputs=gr.Textbox(label="Context-grounded answer", lines=18),
    title="RAG Web Application for ROS2 Support",
    description=(
        "Interactive Gradio interface for querying the ROS2 RAG backend. "
        "The backend retrieves relevant chunks from Qdrant and generates a grounded answer."
    ),
    examples=[
        ["How do I create a ROS2 publisher and subscriber in Python?"],
        ["What is the difference between nav2 costmaps and planners?"],
        ["How do I configure MoveIt2 for a custom robot arm?"],
        ["How can I simulate a robot in Gazebo with ROS2?"],
    ],
)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)