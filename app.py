import gradio as gr
import os
import sys
import tempfile
import traceback
import numpy as np
from pathlib import Path
from PIL import Image

from vector_db import VectorDatabase
from feature_extractor import TreeFeatureExtractor
from vector_normalizer import VectorNormalizer

# ── Import module tien xu ly (optional) ───────────────────
try:
    from preprocessing import TreePreprocessingPipeline
    _PREPROCESS_AVAILABLE = True
    _pipeline = TreePreprocessingPipeline(output_size=(512, 512), segment_method="rembg")
except Exception as _e:
    print(f"[WARN] Khong tai duoc module tien xu ly: {_e}")
    _PREPROCESS_AVAILABLE = False
    _pipeline = None

# ── Đường dẫn tuyệt đối ───────────────────────────────────
APP_DIR = Path(__file__).parent.resolve()
RES_DIR = APP_DIR  

print(f"APP_DIR  = {APP_DIR}")
print(f"RES_DIR  = {RES_DIR}  exists={RES_DIR.exists()}")

# ── Load DB ───────────────────────────────────────────────
print("Dang tai CSDL...")
try:
    normalizer = VectorNormalizer.load(str(APP_DIR / "normalizer.npz"))
    db         = VectorDatabase.load(str(APP_DIR / "vector_db.npz"))
    extractor  = TreeFeatureExtractor(normalizer=normalizer)
    print(f"Tai thanh cong! {len(db)} records.")
except Exception as e:
    print(f"LOI TAI DB: {e}")
    db = extractor = None


# ── Tien xu ly anh truy van ───────────────────────────────
def preprocess_query_image(image_path: str) -> str:
    """
    Tien xu ly anh truy van bang preprocessing module moi
    Tra ve duong dan file tam chua anh da qua xu ly.
    Neu tien xu ly that bai, tra ve path goc.
    """
    if not _PREPROCESS_AVAILABLE or _pipeline is None:
        return image_path

    try:
        result = _pipeline.run(image_path)
        if result.is_valid and result.processed_image is not None:
            # Luu vao file tam de extractor doc
            # Output luon la .png de co the load de dang (chua xu ly background bang thu vien khac)
            tmp = tempfile.NamedTemporaryFile(
                suffix=".png", delete=False, dir=str(APP_DIR)
            )
            # result.processed_image la numpy array
            Image.fromarray(result.processed_image).save(tmp.name)
            tmp.close()
            print(f"[Tien xu ly] Xong: {Path(image_path).name} -> {Path(tmp.name).name}")
            return tmp.name
        else:
            print(f"[Tien xu ly] Anh khong hop le: {result.validation.reason if result.validation else 'Unknown error'}")
            return image_path
    except Exception as e:
        print(f"[Tien xu ly] Loi, dung anh goc: {e}")
        return image_path


def resolve_abs_path(db_path: str) -> Path | None:
    """
    Thu cac vi tri theo thu tu:
      1. Path tuyet doi (neu db_path la absolute)
      2. APP_DIR / db_path  (e.g. res/LoaiCay/file.png)
      3. RES_DIR / rel      (strip 'res'/'tree' prefix)
    Moi buoc thu them duoi .jpg va .jpeg neu khong tim thay ban .png goc.
    """
    p = Path(db_path)

    def _find(candidate: Path) -> Path | None:
        """Thu .jpg truoc (file goc la .jpg), roi moi thu duoi goc va .jpeg/.png."""
        for ext in (".jpg", ".jpeg", candidate.suffix, ".png"):
            alt = candidate.with_suffix(ext)
            if alt.exists():
                return alt
        return None

    # 1. Path tuyet doi
    if p.is_absolute():
        found = _find(p)
        if found:
            return found

    # 2. Relative to APP_DIR
    found = _find(APP_DIR / p)
    if found:
        return found

    # 3. Strip thu muc goc -> RES_DIR
    parts = p.parts
    rel = Path(*parts[1:]) if parts and parts[0].lower() in ("res", "tree") else p
    found = _find(RES_DIR / rel)
    if found:
        return found

    return None


def load_pil(abs_path: Path) -> Image.Image:
    """Doc anh bang PIL (tra ve RGB)."""
    return Image.open(abs_path).convert("RGB")


def make_placeholder(text="Khong tim thay anh") -> Image.Image:
    """Tao anh chu nhat xam lam placeholder."""
    img = Image.new("RGB", (400, 300), color=(45, 45, 60))
    return img


def search_similar_trees(uploaded_path):
    if db is None or extractor is None:
        return "LOI: Chua tai duoc database.", None, None, [], {}
    if uploaded_path is None:
        return "Vui long tai len hoac keo tha anh.", None, None, [], {}

    try:
        # ── Tien xu ly anh ──────────────────────────────────
        processed_path = preprocess_query_image(uploaded_path)
        print(f"[Tien xu ly] Duong dan file tam: {processed_path}")
        preprocessed = processed_path != uploaded_path

        result = extractor.extract(processed_path)

        # BUG 6: Kiểm tra success TRƯỜC khi xóa file tạm
        if not result["success"]:
            if preprocessed:
                Path(processed_path).unlink(missing_ok=True)
            return f"LOI trich xuat: {result['errors']}", None, None, [], {}

        # Load lại các ảnh cho UI
        query_img_pil = load_pil(Path(uploaded_path))
        processed_img_pil = load_pil(Path(processed_path)) if preprocessed else query_img_pil

        # Don dep file tạm sau khi đã load PIL xong
        if preprocessed:
            try:
                Path(processed_path).unlink(missing_ok=True)
            except Exception:
                pass

        feats     = result["features"]   # dict ten -> gia tri
        v_norm    = result.get("vector_normalized")
        query_vec = v_norm if v_norm is not None else result["vector"]
        # Query k+1 de phong truong hop chinh anh truy van nam trong DB
        raw_results = db.query(query_vec, k=6)
        # Loc bo ket qua trung voi anh truy van (distance ~ 0 hoac ten file trung)
        query_fname = Path(uploaded_path).name
        results = [
            r for r in raw_results
            if r["distance"] > 1e-6 and Path(r["image_path"]).name != query_fname
        ][:5]
        # Dat lai thu hang
        for i, r in enumerate(results, start=1):
            r["rank"] = i


        def fmt(v): return f"{v:.4f}"

        preproc_note = "Co (SAM remove-bg + resize)" if preprocessed else "Khong (SAM chua san sang)"
        lines = [
            "=" * 50,
            "[ANH TRUY VAN]",
            f"  File            : {Path(uploaded_path).name}",
            f"  Tien xu ly      : {preproc_note}",
            f"  So chieu         : {result['n_features']} chieu",
            f"  Thoi gian XL     : {result['processing_time_ms']:.0f} ms",
            "",
            "  -- Mau sac (color) --",
            f"  Ty le la xanh    : {fmt(feats.get('color_green_ratio', 0))}",
            f"  Hue mean/std     : {fmt(feats.get('color_h_mean', 0))} / {fmt(feats.get('color_h_std', 0))}",
            f"  Saturation mean  : {fmt(feats.get('color_s_mean', 0))}",
            f"  Brightness mean  : {fmt(feats.get('color_v_mean', 0))}",
            "",
            "  -- Hinh thai (shape) --",
            f"  Aspect ratio     : {fmt(feats.get('shape_aspect_ratio', 0))}",
            f"  Solidity         : {fmt(feats.get('shape_solidity', 0))}",
            f"  Symmetry         : {fmt(feats.get('shape_symmetry', 0))}",
            f"  Crown ratio      : {fmt(feats.get('shape_crown_ratio', 0))}",
            "",
            "  -- Ket cau (texture) --",
            f"  Contrast         : {fmt(feats.get('texture_contrast', 0))}",
            f"  Homogeneity      : {fmt(feats.get('texture_homogeneity', 0))}",
            f"  Energy           : {fmt(feats.get('texture_energy', 0))}",
            f"  Roughness        : {fmt(feats.get('texture_roughness', 0))}",
            "",
            "  -- Tan cay (canopy) --",
            f"  Contour complex  : {fmt(feats.get('canopy_contour_complexity', 0))}",
            f"  Convexity        : {fmt(feats.get('canopy_convexity', 0))}",
            f"  Width mean       : {fmt(feats.get('canopy_width_mean', 0))}",
            "=" * 50,
            "",
            f"[TOP {len(results)} KET QUA TUONG TU]",
        ]

        gallery = []
        paths_info = {
            "query_image_path": str(Path(uploaded_path).resolve()),
            "results": []
        }

        for r in results:
            label = r["label"] or "?"
            dist  = r["distance"]
            rank  = r["rank"]
            db_path = r["image_path"]          # path luu trong DB
            fname = Path(db_path).name
            caption = f"#{rank} {label}  dist={dist:.4f}"

            lines += [
                "",
                f"  Hang {rank}: {fname}",
                f"    Loai       : {label}",
                f"    Khoang cach Euclidean: {dist:.6f}",
                f"    DB path    : {db_path}",
            ]

            paths_info["results"].append({
                "rank": rank,
                "label": label,
                "distance": round(dist, 6),
                "db_path": db_path,
            })

            abs_path = resolve_abs_path(db_path)
            if abs_path:
                try:
                    pil_img = load_pil(abs_path)
                    # Xoay anh ket qua +90 CW (phai) de sua anh bi xoay trai 90 (CCW)
                    # pil_img = pil_img.rotate(-90, expand=True)
                    gallery.append((pil_img, caption))
                except Exception as e:
                    lines.append(f"    LOI doc anh: {e}")
                    gallery.append((make_placeholder(), caption))
            else:
                lines.append("    => Khong tim thay tren dia")
                gallery.append((make_placeholder("Not found"), caption))

        return "\n".join(lines), query_img_pil, processed_img_pil, gallery, paths_info

    except Exception as e:
        tb = traceback.format_exc()
        return f"LOI HE THONG:\n{e}\n\n{tb}", None, None, [], {}


# ── Giao dien Gradio ─────────────────────────────────────
with gr.Blocks(title="Tree Image Search") as demo:
    gr.Markdown(
        """
        # 🌳 He Thong Truy Van Anh Cay
        ### Keo tha anh cay vao o duoi de tim kiem cac cay tuong tu.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                type="filepath",
                label="Anh truy van (keo tha vao day)"
            )
            search_btn  = gr.Button("🔍 Tim kiem", variant="primary")
            output_text = gr.Textbox(label="Log / Thong tin", lines=18)

        with gr.Column(scale=2):
            with gr.Row():
                query_display = gr.Image(
                    label="Anh truy van goc",
                    show_label=True,
                    interactive=False,
                )
                processed_display = gr.Image(
                    label="Anh da tien xu ly (SAM)",
                    show_label=True,
                    interactive=False,
                )
            output_gallery = gr.Gallery(
                label="Top 5 anh tuong tu",
                show_label=True,
                columns=5,
                rows=1,
                object_fit="contain",
                height="auto",
            )

    output_paths = gr.JSON(label="Duong dan anh trong DB")

    search_btn.click(
        fn=search_similar_trees,
        inputs=input_image,
        outputs=[output_text, query_display, processed_display, output_gallery, output_paths],
    )

if __name__ == "__main__":
    print("Khoi dong server...")
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        allowed_paths=[str(RES_DIR)],  # Cho phep Gradio serve anh tu thu muc res
    )
