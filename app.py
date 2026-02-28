import gradio as gr
import os
import shutil

import config
import loader
import retriever


def process_files(files):
    if not files:
        return "No files uploaded.", None

    all_chunks = []
    for file in files:
        try:
            if hasattr(file, "path"):
                src_path = file.path
            else:
                src_path = file

            dst_filename = os.path.basename(src_path)
            dst_path = config.DATA_DIR / dst_filename
            shutil.copy2(src_path, dst_path)

            docs = loader.load_document(str(dst_path))
            chunks = loader.split_documents(docs)
            for chunk in chunks:
                chunk.metadata["source"] = dst_filename
            all_chunks.extend(chunks)
        except Exception as e:
            return (
                f"Error loading {file.name if hasattr(file, 'name') else file}: {e}",
                None,
            )

    if not all_chunks:
        return "No chunks created from documents.", None

    try:
        retriever.initialize_rag(all_chunks)
    except Exception as e:
        return f"Error initializing RAG: {e}", None

    return (
        f"Successfully processed {len(files)} file(s). Created {len(all_chunks)} chunks. Ready to chat!",
        None,
    )


def chat(message, history):
    if not message.strip():
        return history, ""

    if history is None:
        history = []

    if retriever.get_rag_chain() is None:
        history.append({"role": "user", "content": message})
        history.append(
            {"role": "assistant", "content": "Please upload documents first."}
        )
        return history, ""

    try:
        chain = retriever.get_rag_chain()
        response = chain.invoke({"question": message})
        answer = response.content if hasattr(response, "content") else str(response)

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": answer})
    except Exception as e:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": f"Error: {str(e)}"})

    return history, ""


def clear_chat():
    return [], ""


def clear_all():
    retriever.clear_vectorstore()
    return [], "", "Vectorstore cleared. Upload new files to start."


def check_existing_docs():
    if os.path.exists(config.CHROMA_DIR) and os.listdir(config.CHROMA_DIR):
        return True
    return False


with gr.Blocks() as demo:
    gr.Markdown("# Local RAG Chatbot")
    gr.Markdown(f"**LLM:** {config.LLM_MODEL} | **Embedding:** {config.EMBED_MODEL}")
    gr.Markdown(
        "Upload documents and chat with AI powered by Retrieval-Augmented Generation."
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Upload Documents")
            file_input = gr.File(
                label="Upload",
                file_count="multiple",
                file_types=[".pdf", ".txt", ".md", ".csv", ".xlsx", ".xls"],
            )
            process_btn = gr.Button("Process Documents")
            status_output = gr.Textbox(label="Status", lines=3)
            clear_chat_btn = gr.Button("Clear Chat")
            clear_all_btn = gr.Button("Clear All")

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Chat History")
            with gr.Row():
                msg_input = gr.Textbox(
                    label="Message",
                    placeholder="Ask a question about your documents...",
                    lines=1,
                    scale=5,
                )
                submit_btn = gr.Button("Send", scale=1)

    process_btn.click(
        fn=process_files,
        inputs=[file_input],
        outputs=[status_output, file_input],
    )

    submit_btn.click(
        fn=chat,
        inputs=[msg_input, chatbot],
        outputs=[chatbot, msg_input],
    )

    msg_input.submit(
        fn=chat,
        inputs=[msg_input, chatbot],
        outputs=[chatbot, msg_input],
    )

    clear_chat_btn.click(
        fn=clear_chat,
        inputs=[],
        outputs=[chatbot, msg_input],
    )

    clear_all_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[chatbot, msg_input, status_output],
    )

    demo.queue()


if __name__ == "__main__":
    if check_existing_docs():
        print("Found existing vectorstore. Loading...")
        retriever.initialize_rag()

    demo.launch(server_name="0.0.0.0", server_port=7860)
