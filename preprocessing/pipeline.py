"""
pipeline.py — Orchestrates the full preprocessing pipeline
===========================================================

Usage
─────
    from preprocessing import TreePreprocessingPipeline

    pipe = TreePreprocessingPipeline(output_size=(512, 512), segment_method="rembg")
    result = pipe.run("path/to/tree.jpg")
    pipe.save_result(result, "out/processed.jpg")
"""

import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .steps import (
    ValidationResult,
    validate_image,
    fix_orientation,
    normalize_image,
    denoise_image,
    segment_tree,
    crop_to_tree,
    center_and_scale,
    augment_image,
)
from .utils import load_image, save_image, save_mask, visualize_pipeline


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """
    Full output from TreePreprocessingPipeline.run().

    Attributes
    ──────────
    source_path      : Original input file path.
    is_valid         : Whether the image passed validation.
    validation        : Detailed ValidationResult object.
    processed_image  : Final preprocessed RGB image (uint8, fixed size).
    mask             : Binary tree mask (uint8, 255 = tree). May be None if
                       segmentation was disabled.
    stage_images     : Dict of intermediate stage images (for visualization).
    metadata         : Dict with bbox, coverage, and other computed stats.
    """
    source_path:     str
    is_valid:        bool
    validation:      Optional[ValidationResult]
    processed_image: Optional[np.ndarray]
    mask:            Optional[np.ndarray]
    stage_images:    Dict[str, Optional[np.ndarray]] = field(default_factory=dict)
    metadata:        Dict = field(default_factory=dict)

    # ── Convenience properties ─────────────────────────────────────────────

    @property
    def tree_coverage(self) -> float:
        """Fraction of the final image occupied by the tree (0–1)."""
        if self.mask is None:
            return 0.0
        return float((self.mask > 0).mean())

    def summary(self) -> str:
        lines = [
            f"Source      : {self.source_path}",
            f"Valid       : {self.is_valid}",
            f"Validation  : {self.validation.reason if self.validation else 'N/A'}",
        ]
        if self.processed_image is not None:
            h, w = self.processed_image.shape[:2]
            lines.append(f"Output size : {w}×{h}")
        lines.append(f"Tree coverage: {self.tree_coverage:.2%}")
        if self.metadata:
            for k, v in self.metadata.items():
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class TreePreprocessingPipeline:
    """
    End-to-end image preprocessing pipeline for tree similarity search.

    Steps
    ─────
    1. Load image
    2. Validate (filter non-tree / illustration)
    3. Fix orientation (EXIF)
    4. Normalize (resize + CLAHE)
    5. Denoise
    6. Segment tree (rembg / GrabCut)
    7. Crop to tree bounding box
    8. Center & scale to fixed canvas
    9. Augment (optional)

    Args
    ────
    output_size         : Final (width, height) for all outputs.
    segment_method      : ``"rembg"`` (best) or ``"grabcut"`` (no GPU needed).
    denoise_method      : ``"nlmeans"``, ``"bilateral"``, ``"gaussian"``,
                          or ``"none"``.
    normalize_pixels    : Return float32 tensor-ready output when True.
    augment             : Apply random augmentation.
    skip_validation     : Bypass the validation step (process all images).
    white_background    : Fill background with white; else keep original BG.
    verbose             : Print progress messages.
    """

    def __init__(
        self,
        output_size:      Tuple[int, int] = (512, 512),
        segment_method:   str = "rembg",
        denoise_method:   str = "nlmeans",
        normalize_pixels: bool = False,
        augment:          bool = False,
        skip_validation:  bool = False,
        white_background: bool = True,
        verbose:          bool = True,
    ):
        self.output_size      = output_size
        self.segment_method   = segment_method
        self.denoise_method   = denoise_method
        self.normalize_pixels = normalize_pixels
        self.augment          = augment
        self.skip_validation  = skip_validation
        self.white_background = white_background
        self.verbose          = verbose

        # Shared rembg session (loaded once, reused for all images in a batch)
        self._rembg_session = None
        if segment_method == "rembg":
            try:
                from rembg import new_session
                self._rembg_session = new_session("u2net")
                self._log("rembg session (u2net) loaded.")
            except Exception as e:
                warnings.warn(
                    f"Could not create rembg session ({e}). "
                    "Will try per-call or fall back to GrabCut."
                )

    # ── Logging ────────────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[Pipeline] {msg}")

    # ── Single-image run ───────────────────────────────────────────────────

    def run(self, image_path: str) -> PipelineResult:
        """
        Process a single image through the full pipeline.

        Args:
            image_path: Path to the input image (jpg/png).

        Returns:
            PipelineResult with all intermediate stages and final output.
        """
        self._log(f"Processing: {os.path.basename(image_path)}")
        stage_images: Dict[str, Optional[np.ndarray]] = {}

        # ── 1. Load ──────────────────────────────────────────────────────────
        try:
            pil_img = Image.open(image_path)
            image   = np.array(pil_img.convert("RGB"))
        except Exception as e:
            return PipelineResult(
                source_path=image_path, is_valid=False,
                validation=None, processed_image=None, mask=None,
                stage_images={}, metadata={"error": str(e)},
            )

        stage_images["1_original"] = image.copy()

        # ── 2. Validate ───────────────────────────────────────────────────────
        if not self.skip_validation:
            val = validate_image(image)
            self._log(f"  Validation: {val.reason}")
            if not val.is_valid:
                return PipelineResult(
                    source_path=image_path, is_valid=False,
                    validation=val, processed_image=None, mask=None,
                    stage_images=stage_images,
                )
        else:
            val = ValidationResult(True, "validation skipped")

        # ── 3. Fix orientation ────────────────────────────────────────────────
        image = fix_orientation(image, pil_img)
        stage_images["2_oriented"] = image.copy()

        # ── 4. Normalize ──────────────────────────────────────────────────────
        image = normalize_image(image, target_size=self.output_size, normalize_pixels=False)
        stage_images["3_normalized"] = image.copy()

        # ── 5. Denoise ────────────────────────────────────────────────────────
        image = denoise_image(image, method=self.denoise_method)
        stage_images["4_denoised"] = image.copy()

        # ── 6. Segment ────────────────────────────────────────────────────────
        self._log(f"  Segmenting with '{self.segment_method}' …")
        mask, segmented = segment_tree(
            image,
            method=self.segment_method,
            rembg_session=self._rembg_session,
            white_background=self.white_background,
        )
        stage_images["5_mask"]      = mask.copy()
        stage_images["6_segmented"] = segmented.copy()

        # ── 7. Crop ───────────────────────────────────────────────────────────
        cropped, cropped_mask, bbox = crop_to_tree(segmented, mask)
        stage_images["7_cropped"] = cropped.copy()

        # ── 8. Center & scale ─────────────────────────────────────────────────
        final_img, final_mask = center_and_scale(
            cropped, cropped_mask, output_size=self.output_size
        )
        stage_images["8_final"] = final_img.copy()

        # ── 9. Augment (optional) ─────────────────────────────────────────────
        if self.augment:
            final_img, final_mask = augment_image(final_img, final_mask)
            stage_images["9_augmented"] = final_img.copy()

        # ── Compute metadata ──────────────────────────────────────────────────
        orig_h, orig_w = np.array(pil_img).shape[:2]
        bx, by, bw, bh = bbox
        tree_coverage = float((final_mask > 0).mean())

        metadata = {
            "original_size":   f"{orig_w}×{orig_h}",
            "output_size":     f"{self.output_size[0]}×{self.output_size[1]}",
            "bbox_x":          bx,
            "bbox_y":          by,
            "bbox_w":          bw,
            "bbox_h":          bh,
            "tree_coverage":   round(tree_coverage, 4),
            "validation_issues": val.issues,
        }

        self._log(f"  Done. Tree coverage: {tree_coverage:.2%}")

        return PipelineResult(
            source_path=image_path,
            is_valid=True,
            validation=val,
            processed_image=final_img,
            mask=final_mask,
            stage_images=stage_images,
            metadata=metadata,
        )

    # ── Batch run ──────────────────────────────────────────────────────────

    def run_batch(
        self,
        image_paths: List[str],
        output_dir:  Optional[str] = None,
        base_dir:    Optional[str] = None,
        save_masks:  bool = False,
        save_viz:    bool = False,
    ) -> List[PipelineResult]:
        """
        Process a list of images and optionally save results to disk.

        Args:
            image_paths: List of input image paths.
            output_dir:  If given, saves processed images here.
            base_dir:    Root directory of the input scan. When provided,
                         the relative sub-path is mirrored inside output_dir
                         (e.g. base_dir/cat1/img.jpg → output_dir/cat1/img.jpg).
                         When None, all images are written flat into output_dir.
            save_masks:  Also save the binary masks to ``output_dir/masks/``.
            save_viz:    Also save the pipeline visualization strip.

        Returns:
            List of PipelineResult (one per input image).
        """
        results = []
        n = len(image_paths)
        for i, path in enumerate(image_paths, start=1):
            self._log(f"[{i}/{n}] {os.path.relpath(path, base_dir) if base_dir else os.path.basename(path)}")

            # ── Per-image try/except: one bad image must NOT kill the batch ──
            try:
                result = self.run(path)
            except Exception as exc:
                warnings.warn(f"[Pipeline] Unhandled error on {path}: {exc}")
                result = PipelineResult(
                    source_path=path,
                    is_valid=False,
                    validation=None,
                    processed_image=None,
                    mask=None,
                    stage_images={},
                    metadata={"error": str(exc)},
                )

            results.append(result)

            if output_dir and result.is_valid and result.processed_image is not None:
                abs_path = os.path.abspath(path)
                fname    = os.path.basename(abs_path)

                # Mirror sub-directory structure when base_dir is given
                if base_dir:
                    rel = os.path.relpath(abs_path, os.path.abspath(base_dir))
                    out_img_path = os.path.join(output_dir, rel)
                else:
                    out_img_path = os.path.join(output_dir, fname)

                os.makedirs(os.path.dirname(out_img_path), exist_ok=True)
                save_image(result.processed_image, out_img_path)

                if save_masks and result.mask is not None:
                    base = os.path.splitext(fname)[0]
                    mask_path = os.path.join(output_dir, "masks", f"{base}_mask.png")
                    save_mask(result.mask, mask_path)

                if save_viz:
                    base = os.path.splitext(fname)[0]
                    viz_path = os.path.join(output_dir, "viz", f"{base}_viz.jpg")
                    visualize_pipeline(result.stage_images, output_path=viz_path)

        valid   = sum(1 for r in results if r.is_valid)
        invalid = n - valid
        self._log(f"Batch complete: {valid} valid, {invalid} invalid out of {n} images.")
        return results

    # ── Save helpers ───────────────────────────────────────────────────────

    def save_result(
        self,
        result: PipelineResult,
        output_path: str,
        save_mask_path: Optional[str] = None,
        save_viz_path:  Optional[str] = None,
    ) -> None:
        """
        Save the pipeline result to disk.

        Args:
            result:          PipelineResult from run().
            output_path:     Path to save the processed image.
            save_mask_path:  Optional path to save the binary mask.
            save_viz_path:   Optional path to save the visualization strip.
        """
        if not result.is_valid or result.processed_image is None:
            print(f"[Pipeline] Cannot save invalid result for {result.source_path}")
            return

        save_image(result.processed_image, output_path)
        self._log(f"Saved processed image → {output_path}")

        if save_mask_path and result.mask is not None:
            save_mask(result.mask, save_mask_path)
            self._log(f"Saved mask → {save_mask_path}")

        if save_viz_path:
            visualize_pipeline(result.stage_images, output_path=save_viz_path)
