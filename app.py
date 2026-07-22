from __future__ import annotations

from pathlib import Path
import sys
from collections import Counter
from hashlib import md5
from typing import Any, Dict, List

import cv2
import numpy as np
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from infer_pipeline import apply_foreground_mask, load_maskrcnn, load_unet, run_maskrcnn, run_unet_mask
from utils.common import get_device
from utils.visualization import colorize_semantic_mask, draw_instance_predictions

st.set_page_config(page_title="Agri Vision Demo", page_icon="🌿", layout="wide")

CUSTOM_CSS = '''
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
.app-card {
    background: linear-gradient(135deg, rgba(40,120,80,0.10), rgba(20,40,20,0.02));
    border: 1px solid rgba(90,150,110,0.25);
    border-radius: 18px;
    padding: 18px 18px 12px 18px;
    margin-bottom: 14px;
}
.result-chip {
    display: inline-block;
    padding: 0.28rem 0.7rem;
    border-radius: 999px;
    background: rgba(18, 130, 70, 0.10);
    border: 1px solid rgba(18,130,70,0.22);
    margin: 0.15rem 0.25rem 0.15rem 0;
    font-size: 0.95rem;
}
.info-note {
    padding: 0.55rem 0.8rem;
    border-left: 3px solid rgba(18,130,70,0.8);
    background: rgba(18,130,70,0.08);
    border-radius: 8px;
    margin: 0.25rem 0 0.6rem 0;
}
</style>
'''
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

UNET_CKPT = "outputs/unet/best_unet.pth"
MASKRCNN_CKPT = "outputs/maskrcnn/best_maskrcnn.pth"

CLASS_NAMES = [
    "background",
    "data",
    "bell pepper",
    "carrot",
    "garlic",
    "mangosteen",
    "pineapple",
    "pitaya",
    "strawberry",
    "tomato",
]

ITEM_GROUPS = {
    "data": "Khác",
    "bell pepper": "Quả",
    "carrot": "Củ",
    "garlic": "Củ",
    "mangosteen": "Quả",
    "pineapple": "Quả",
    "pitaya": "Quả",
    "strawberry": "Quả",
    "tomato": "Quả",
}

ITEM_INFO = {
    "data": "Lớp tổng quát hoặc nhiễu từ dataset cũ, không phải tên nông sản cụ thể.",
    "bell pepper": "là loại quả họ Cà, giàu dinh dưỡng, ít calo và chứa nhiều vitamin C, A, cùng các chất chống oxy hóa, hỗ trợ tăng cường miễn dịch, tốt cho mắt, làn da và tim mạch. Với hương vị giòn, ngọt và không cay, ớt chuông thường có các màu đỏ, vàng, cam (chín) hoặc xanh (chưa chín), được sử dụng sống hoặc chế biến.",
    "carrot": "là loại rau ăn củ phổ biến, nổi tiếng giàu beta-carotene (tiền tố vitamin A), vitamin K1, kali và chất xơ, rất tốt cho mắt, hệ miễn dịch và tim mạch. Loại củ này có màu cam phổ biến, ngoài ra còn có màu vàng, trắng, tía, có vị ngọt thanh, thích hợp cho người giảm cân, ăn kiêng và chăm sóc da.",
    "garlic": "là loài thực vật thuộc họ Hành, nguồn gốc Trung Á, nổi tiếng với công dụng làm gia vị ẩm thực và dược liệu quý. Tỏi giàu hợp chất hữu cơ sulfur (đặc biệt là allicin), vitamin B6, C, mangan, mang đặc tính kháng khuẩn, kháng virus, chống oxy hóa, hỗ trợ tim mạch và tăng cường miễn dịch.",
    "mangosteen": "là loại cây ăn trái nhiệt đới Đông Nam Á, được mệnh danh là nữ hoàng trái cây với vỏ dày màu tím, bên trong là múi trắng mềm, vị ngọt thanh và thơm. Quả chứa nhiều dưỡng chất, chất chống oxy hóa (xanthones) tốt cho da và tiêu hóa, thường được thu hoạch vào mùa hè tại Việt Nam.",
    "pineapple": "là loại cây ăn quả nhiệt đới thân thảo, nguồn gốc Nam Mỹ, nổi tiếng với vị chua ngọt, hương thơm đặc trưng và giàu dinh dưỡng. Dứa chứa nhiều vitamin C, bromelain (enzyme tiêu hóa) và khoáng chất, hỗ trợ miễn dịch, tiêu hóa tốt và có lợi cho sức khỏe tim mạch.",
    "pitaya": "là loại trái cây nhiệt đới nổi tiếng thuộc họ Xương rồng (Cactaceae), có nguồn gốc từ Trung Mỹ và Mexico. Với vỏ màu hồng đỏ và thịt ruột trắng hoặc đỏ (chứa hạt nhỏ đen), thanh long mang lại vị ngọt thanh, giòn dịu, giàu vitamin, chất chống oxy hóa và rất tốt cho sức khỏe.",
    "strawberry": "là loại trái cây ôn đới thuộc họ Hoa hồng, giàu vitamin C, kali, chất chống oxy hóa và ít calo. Quả có hương vị chua ngọt, hình trái tim đặc trưng, mang ý nghĩa tượng trưng cho tình yêu và hạnh phúc. Dâu tây rất tốt cho tim mạch, kiểm soát đường huyết (chỉ số GI thấp ~40) và giúp làm đẹp da.",
    "tomato": "là loại rau ăn quả phổ biến, giàu dinh dưỡng (Vitamin A, C, K, B6, lycopene) và có nguồn gốc từ Nam Mỹ. Quả có vị chua ngọt, màu đỏ/vàng, dùng phổ biến trong ẩm thực, giúp chống oxy hóa, làm đẹp da, tốt cho tim mạch và thị lực. Cây ưa ẩm, dễ trồng, cho trái quanh năm.",
}


@st.cache_resource(show_spinner=False)
def cached_models(device_name: str):
    device = get_device(device_name)
    unet_model, task_mode, _, unet_img_size = load_unet(UNET_CKPT, device)
    maskrcnn_model, _ = load_maskrcnn(MASKRCNN_CKPT, device)
    return unet_model, task_mode, int(unet_img_size), maskrcnn_model, device


def infer_group(name: str) -> str:
    return ITEM_GROUPS.get((name or "").strip().lower(), "Khác")


def infer_info(name: str) -> str:
    if name in ITEM_INFO:
        return ITEM_INFO[name]
    group = infer_group(name)
    return f"Thuộc nhóm {group.lower()}. Chưa có mô tả chi tiết cho lớp này."


def file_hash(file_bytes: bytes) -> str:
    return md5(file_bytes).hexdigest()


def ensure_state():
    defaults = {"active_image_hash": None, "active_image_name": None}
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def summarize_unet(semantic_mask: np.ndarray, class_names: List[str]) -> Dict[str, Any]:
    unique_ids, counts = np.unique(semantic_mask, return_counts=True)
    total = int(semantic_mask.size)
    present = []
    for cls_id, count in zip(unique_ids, counts):
        cls_id = int(cls_id)
        if cls_id == 0:
            continue
        name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
        if name == "data":
            continue
        present.append(
            {
                "class_id": cls_id,
                "class_name": name,
                "pixels": int(count),
                "ratio": float(count) / max(total, 1),
                "group": infer_group(name),
            }
        )
    present.sort(key=lambda x: x["pixels"], reverse=True)
    dominant = present[0]["class_name"] if present else "Không xác định"
    dominant_group = present[0]["group"] if present else "Khác"
    return {"dominant_name": dominant, "dominant_group": dominant_group, "present": present}


def summarize_maskrcnn(boxes, labels, scores, class_names: List[str], score_thr: float) -> Dict[str, Any]:
    kept = scores >= score_thr if len(scores) else np.zeros((0,), dtype=bool)
    kept_labels = labels[kept] if len(labels) else np.zeros((0,), dtype=np.int64)
    per_class = Counter()
    for cls_id in kept_labels.tolist():
        name = class_names[int(cls_id)] if int(cls_id) < len(class_names) else str(int(cls_id))
        if name == "data":
            continue
        per_class[name] += 1
    class_rows = []
    for name, count in per_class.most_common():
        class_rows.append({"name": name, "group": infer_group(name), "count": int(count), "info": infer_info(name)})
    main_name = class_rows[0]["name"] if class_rows else "Không xác định"
    main_group = class_rows[0]["group"] if class_rows else "Khác"
    return {"object_count": int(sum(per_class.values())), "class_rows": class_rows, "main_name": main_name, "main_group": main_group}


def final_classification(mask_summary: Dict[str, Any], unet_summary: Dict[str, Any]) -> Dict[str, str]:
    if mask_summary["object_count"] > 0:
        name = mask_summary["main_name"]
        group = mask_summary["main_group"]
        source = "Mask R-CNN"
    else:
        name = unet_summary["dominant_name"]
        group = unet_summary["dominant_group"]
        source = "U-Net"
    return {"name": name, "group": group, "info": infer_info(name), "source": source}


def render_results(
    image_rgb: np.ndarray,
    unet_overlay: np.ndarray,
    maskrcnn_overlay: np.ndarray,
    final_result: Dict[str, str],
    mask_summary: Dict[str, Any],
    unet_summary: Dict[str, Any],
    device_type: str,
    use_unet_first: bool,
):
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Device", device_type)
    m2.metric("Loại", final_result["group"])
    m3.metric("Tên", final_result["name"])
    m4.metric("Số lượng", mask_summary["object_count"])

    mode_text = (
        "Dùng U-Net để phân vùng đối tượng trước, sau đó Mask R-CNN tách các đối tượng"
        if use_unet_first
        else "Mask R-CNN tách các đối tượng trực tiếp trên ảnh gốc"
    )
    st.markdown(f"<div class='info-note'><b>Chế độ hiện tại:</b> {mode_text}</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1, 1.1])
    with c1:
        st.subheader("Ảnh gốc")
        st.image(image_rgb, use_container_width=True)
    with c2:
        st.subheader("Kết quả phân vùng của U-Net")
        st.image(unet_overlay, use_container_width=True)
    with c3:
        st.subheader("Kết quả phân vùng của Mask R-CNN")
        st.image(maskrcnn_overlay, use_container_width=True)

    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.subheader("Kết quả phân loại")
    st.markdown(f"<span class='result-chip'>Nhóm: {final_result['group']}</span>", unsafe_allow_html=True)
    st.markdown(f"<span class='result-chip'>Tên: {final_result['name']}</span>", unsafe_allow_html=True)
    st.markdown(f"<span class='result-chip'>Số lượng: {mask_summary['object_count']}</span>", unsafe_allow_html=True)

    if mask_summary["class_rows"]:
        first_row = mask_summary["class_rows"][0]
        st.markdown(f"**Thông tin:** {first_row['info']}")
        if len(mask_summary["class_rows"]) > 1:
            others = ", ".join([f"{row['name']} ({row['count']})" for row in mask_summary["class_rows"][1:]])
            st.markdown(f"**Phát hiện thêm:** {others}")
    else:
        st.markdown(f"**Thông tin:** {final_result['info']}")

    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Kết quả phân tích")
    if unet_summary["present"]:
        st.dataframe(
            [
                {
                    "Tên": row["class_name"],
                    "Nhóm": row["group"],
                    "Độ tin cậy": round(row["ratio"] * 100, 2),
                    "Tỉ lệ vùng": round(row["ratio"], 4),
                    "Pixels": row["pixels"],
                }
                for row in unet_summary["present"]
            ],
            use_container_width=True,
        )
    else:
        st.info("U-Net chưa tìm thấy vùng foreground rõ ràng.")


ensure_state()

st.title("🌿 Agri Vision Studio")
st.caption("Hệ thống phân loại sản phẩm nông nghiệp.")

with st.sidebar:
    st.header("Cấu hình")
    auto_device = "cuda" if torch.cuda.is_available() else "cpu"
    device_option = st.selectbox("Device", [auto_device, "cpu", "cuda"], index=0)
    st.markdown("### Threshold")
    score_thr = st.slider("Score threshold", min_value=0.1, max_value=0.95, value=0.5, step=0.05)
    mask_thr = st.slider("Mask threshold", min_value=0.1, max_value=0.95, value=0.5, step=0.05)
    use_unet_first = st.checkbox(
        "Dùng U-Net để phân vùng đối tượng trước rồi Mask R-CNN tách các đối tượng",
        value=True,
        help="Bật = U-Net tạo vùng foreground trước, sau đó Mask R-CNN chạy trên ảnh đã được U-Net phân vùng. Tắt = Mask R-CNN chạy trực tiếp trên ảnh gốc.",
    )

st.markdown('<div class="app-card">', unsafe_allow_html=True)
st.subheader("Chọn ảnh để phân loại")
uploaded_file = st.file_uploader("Tải ảnh nông sản", type=["jpg", "jpeg", "png", "bmp", "webp"], key="image_uploader")
run_clicked = st.button("Phân loại", type="primary", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is None:
    st.info("Tải một ảnh lên, sau đó bấm “Phân loại”.")
    st.stop()

file_bytes = uploaded_file.getvalue()
current_hash = file_hash(file_bytes)
is_current_image_active = st.session_state.active_image_hash == current_hash

if run_clicked:
    st.session_state.active_image_hash = current_hash
    st.session_state.active_image_name = uploaded_file.name
    is_current_image_active = True

if not is_current_image_active:
    st.info("Ảnh mới đã được tải lên. Nhấn nút “Phân loại” để hiển thị kết quả cho ảnh này.")
    st.image(file_bytes, caption=uploaded_file.name, use_container_width=True)
    st.stop()

image_bgr = cv2.imdecode(np.asarray(bytearray(file_bytes), dtype=np.uint8), cv2.IMREAD_COLOR)
if image_bgr is None:
    st.error("Không đọc được ảnh tải lên.")
    st.stop()
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

try:
    unet_model, task_mode, unet_img_size, maskrcnn_model, device = cached_models(device_option)
except Exception as exc:
    st.error(f"Lỗi load model: {exc}")
    st.stop()

with st.spinner("Đang phân loại và phân vùng..."):
    semantic_mask = run_unet_mask(unet_model, image_rgb, int(unet_img_size), device, task_mode, mask_thr)
    maskrcnn_input = apply_foreground_mask(image_rgb, semantic_mask) if use_unet_first else image_rgb.copy()
    boxes, labels, scores, masks = run_maskrcnn(maskrcnn_model, maskrcnn_input, device)

    if len(scores):
        boxes = np.asarray(boxes)
        labels = np.asarray(labels)
        scores = np.asarray(scores)
        masks = np.asarray(masks)

    unet_overlay = colorize_semantic_mask(semantic_mask)
    maskrcnn_overlay = draw_instance_predictions(
        image_rgb.copy(),
        boxes=boxes,
        labels=labels,
        scores=scores,
        masks=masks,
        class_names=CLASS_NAMES,
        score_thr=score_thr,
    )

unet_summary = summarize_unet(semantic_mask, CLASS_NAMES)
mask_summary = summarize_maskrcnn(boxes, labels, scores, CLASS_NAMES, score_thr)
final_result = final_classification(mask_summary, unet_summary)

render_results(
    image_rgb=image_rgb,
    unet_overlay=unet_overlay,
    maskrcnn_overlay=maskrcnn_overlay,
    final_result=final_result,
    mask_summary=mask_summary,
    unet_summary=unet_summary,
    device_type=device.type,
    use_unet_first=use_unet_first,
)
