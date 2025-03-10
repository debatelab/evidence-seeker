import gradio as gr
import random
import dotenv

from evidence_seeker import EvidenceSeeker, describe_result
from evidence_seeker.retrieval import RetrievalConfig

EMBEDDING_PATH = "../TMP/APUZ/storage/index"

EXAMPLE_INPUTS = [
    "Die Osterweiterung hat die EU-Institutionen nachhaltig geschwächt.",
    "In den knapp 70 Jahren seit ihrer Gründung hat es in der Bundeswehr immer wieder rechtsextremistische Vorfälle gegeben.",
    "In der Bundeswehr gibt es keinen politischen Extremismus.",
    "Die BRICS-Staaten sorgen für eine Veränderung der westlich geprägten Weltordnung.",
    "Die Genfer Konventionen sind oft hinter ihrem Anspruch, die Zivilbevölkerung zu schützen, zurückgeblieben.",
    "Die Anzahl hybrider Kriege hat zugenommen.",
    "Es ist für Deutschland wirtschaftlich ein Nachteil, dass viele Frauen in Teilzeit arbeiten.",
    "Premier Modi hat Putin als seinen Freund bezeichnet.",
    "Eine Minderheit der deutschen Bevölkerung befürwortet einen autoritären deutschen Staat.",
]

def setup(index_path : str) -> EvidenceSeeker:
    # Uses endpoints
    dotenv.load_dotenv()
    retrieval_config = RetrievalConfig(
        index_persist_path=index_path,
    )
    evidence_seeker = EvidenceSeeker(retrieval_config=retrieval_config)
    return evidence_seeker

EVIDENCE_SEEKER = setup(EMBEDDING_PATH)

def draw_example(examples : list[str]) -> str:
    random.shuffle(examples)
    return examples[0]

async def check(statement : str) -> str:
    if EVIDENCE_SEEKER is None:
        return f"Kein EvidenceSeeker initialisiert um '{statement}' zu prüfen."
    result = await EVIDENCE_SEEKER(statement)
    md = describe_result(statement, result)
    return md 

with gr.Blocks(title="EvidenceSeeker") as demo:
    gr.Markdown("""
        # 🕵️ EvidenceSeeker DemoApp
        Gib eine Aussage in das Textfeld ein und lass sie durch den EvidenceSeeker prüfen:
    """
    )
    examples = gr.State(EXAMPLE_INPUTS)

    with gr.Row():
        statement = gr.Textbox(label="Zu prüfende Aussage:", elem_id="statement", scale=10,lines=3)
        with gr.Column(scale=1):
            example_btn = gr.Button("Zufälliges Beispiel", elem_id="example_btn")
            check_btn = gr.Button("Prüfe Aussage", interactive=False)
    results = gr.Markdown(value="", show_copy_button=True)

    example_btn.click(fn=draw_example, inputs=examples, outputs=statement)
    check_btn.click(check, statement, results)
    statement.change(lambda s: (gr.Button("Prüfe Aussage", interactive=False) if s.strip() == "" else gr.Button("Prüfe Aussage", interactive=True)), statement, check_btn)
    demo.launch()
