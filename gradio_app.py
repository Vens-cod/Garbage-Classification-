"""Gradio 教学版 - 登录 + 模型训练 + 推理功能"""

from __future__ import annotations
import locale
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
import csv
import tempfile
from typing import Any
import gradio as gr
import torch
from werkzeug.security import check_password_hash, generate_password_hash
from config import AppConfig
from db import Database

_CFG = AppConfig.load()
_EPOCH_PROGRESS_RE = re.compile(r"Epoch\s+(\d+)\s*/\s*(\d+)|Epoch\s+(\d+)/(\d+)")
_TRAIN_NUM_WORKERS = 2
_TRAIN_FORCE_CPU = False
_TRAIN_RESNET50_PRETRAINED = True
_CURRENT_MODEL_KEY = "current_model"
_DB: Database | None = None

APP_CSS = """
#login-wrap{
    max-width: 420px;
    margin: 10vh auto 0 auto;
    padding: 18px;
    border: 1px solid rgb(0,0,0,.08);
    border-radius: 12px;
    background: rgb(255,255,255,.92);
    box-shadow: 0 8px 30px rgb(0,0,0,.08);
}
main-panel{
    max-width: 1180px;
    margin: 0 auto;
    padding: 14px 12px 24px 12px;
}
main-panel::before{
    content: "";
    position: fixed;
    inset: 0;
    z-index: -1;
    background: radial-gradient(1200px 800px at 10% 10%, rgb(59,130,246,.08), transparent 55%),
                radial-gradient(1200px 800px at 90% 0%, rgb(168,85,247,.08), transparent 55%),
                #f8fafc;
}
"""


def _project_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _model_output_dir(model_name: str) -> str:
    return os.path.join(_project_root(), "models", model_name)


def _to_relative_project_path(path: str) -> str:
    abs_path = os.path.abspath(path)
    root = _project_root()
    try:
        rel_path = os.path.relpath(abs_path, root)
    except ValueError:
        return abs_path
    return rel_path.replace("/", os.sep)


def _items_to_rows(items: list[dict[str, Any]], fields: list[str]) -> list[list[Any]]:
    return [[item.get(field) for field in fields] for item in (items or [])]


def _models_to_rows(items: list[dict[str, Any]]) -> list[list[Any]]:
    return _items_to_rows(items, ["name", "size_mb", "num_classes", "epoch", "val_acc", "updated_at", "is_current"])


def _get_db() -> Database:
    global _DB
    if _DB is None:
        _DB = Database(_CFG.db_url())
        _DB.init_tables()
    return _DB


def _row_to_dict(row: Any) -> dict[str, Any]:
    if row is None:
        return {}
    if isinstance(row, dict):
        return dict(row)
    if hasattr(row, "keys"):
        return {key: row[key] for key in row.keys()}
    return {}


def _list_models_from_db() -> list[dict[str, Any]]:
    db = _get_db()
    current_name = (db.get_setting(_CURRENT_MODEL_KEY, "") or "").strip()
    items: list[dict[str, Any]] = []
    for row in db.list_models():
        item = _row_to_dict(row)
        item["is_current"] = item.get("name") == current_name
        items.append(item)
    return items


def _get_current_model_name_from_db() -> str | None:
    db = _get_db()
    current_name = (db.get_setting(_CURRENT_MODEL_KEY, "") or "").strip()
    items = _list_models_from_db()
    valid_names = {str(item.get("name") or "").strip() for item in items if item.get("name")}
    if current_name and current_name in valid_names:
        return current_name
    if current_name:
        db.set_setting(_CURRENT_MODEL_KEY, "")
    if not items:
        return None
    return items[0].get("name")


def _register_model_record(checkpoint_path: str, fallback_arch: str) -> None:
    db = _get_db()
    checkpoint_path = os.path.abspath(checkpoint_path)
    if not os.path.exists(checkpoint_path):
        return
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    arch = str(checkpoint.get("arch") or fallback_arch or "")
    class_names = checkpoint.get("class_names") or []
    num_classes = checkpoint.get("num_classes")
    if num_classes is None and isinstance(class_names, list):
        num_classes = len(class_names)
    epoch = checkpoint.get("epoch")
    val_acc = checkpoint.get("val_acc")
    size_mb = round(os.path.getsize(checkpoint_path) / (1024 * 1024), 2)
    now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    name = os.path.basename(checkpoint_path)
    relative_path = _to_relative_project_path(checkpoint_path)

    if db.check_model_exists(name):
        db.update_model(name, relative_path, arch, size_mb, num_classes, epoch, val_acc, now_text)
    else:
        db.create_model(name, relative_path, arch, size_mb, num_classes, epoch, val_acc, now_text, now_text)


def _refresh_weights():
    try:
        return [item["name"] for item in _list_models_from_db()]
    except Exception:
        return []


def _get_current_weight():
    try:
        return _get_current_model_name_from_db()
    except Exception:
        weights = _refresh_weights()
        return weights[0] if weights else None


def _models_list(auth):
    try:
        items = _list_models_from_db()
        current = _get_current_model_name_from_db()
        return items, current or "", f"共 {len(items)} 个模型"
    except Exception as exc:
        return [], None, f"加载模型列表失败: {exc}"


def _model_panel_state(items, cur, msg, choices):
    return _models_to_rows(items), cur or "", msg, gr.update(choices=choices, value=cur or None)


# ========== 推理功能：图片保存与表格工具 ==========

def _record_image_dir() -> str:
    """返回识别快照的保存目录（项目根目录/records/images/）"""
    return os.path.join(_project_root(), "records", "images")


def _save_record_image(image) -> str:
    """
    将 PIL Image 保存到 records/images/ 目录，返回相对路径。
    参数:
        image: PIL Image 对象
    返回:
        相对于项目根目录的图片路径（如 records/images/record_20250608_120101_123456.png）
    """
    if image is None:
        return ""
    os.makedirs(_record_image_dir(), exist_ok=True)
    filename = f"record_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
    abs_path = os.path.join(_record_image_dir(), filename)
    image.save(abs_path)
    return _to_relative_project_path(abs_path)


def _records_to_rows(items: list[dict[str, Any]]) -> list[list[Any]]:
    """将记录字典列表转换为 Gradio Dataframe 需要的二维列表，与表格表头对应"""
    return _items_to_rows(
        items,
        ["id", "label", "confidence", "model", "weight", "user", "created_at"]
    )


# ========== 推理功能：识别回调 ==========

def _recognize(auth, image, weight_name):
    """
    垃圾分类识别回调（当前为占位示例，选做可接入真实模型）
    参数:
        auth: 用户认证信息（gr.State）
        image: PIL Image 对象
        weight_name: 选中的权重文件名
    返回:
        preview: 原图（用于预览组件）
        result: JSON 格式的识别结果字典
        msg: 提示字符串
        topk: Top-K 表格数据（二维列表）
    """
    if image is None:
        return None, None, "请先上传图片", []

    weight_name = (weight_name or _get_current_weight() or "").strip()
    model_name = ""

    if weight_name:
        try:
            model_name = _get_db().get_model_arch(weight_name)
        except Exception:
            model_name = ""
    model_name = model_name or "unknown"

    # 占位识别结果（示例数据）
    result = {
        "label": "可回收物",
        "confidence": 0.95,
        "weight": weight_name,
        "model": model_name,
        "message": "这里是占位识别结果, 等待学生接入真实模型。"
    }

    topk = [
        ["可回收物", 0.95],
        ["厨余垃圾", 0.03],
        ["其他垃圾", 0.02],
    ]

    # 写入数据库
    try:
        _get_db().create_record(
            thumb=_save_record_image(image),
            label=result["label"],
            confidence=float(result["confidence"]),
            elapsed=0.0,
            model=model_name,
            weight=weight_name,
            user=str((auth or {}).get("username") or ""),
            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
    except Exception as exc:
        result["message"] = f'{result["message"]} 记录保存失败: {exc}'

    return image, result, "识别完成 (示例数据)", topk


# ========== 推理功能：辅助函数（路径转绝对）==========
def _to_absolute_project_path(rel_path: str) -> str:
    """
    将相对于项目根目录的路径转换为绝对路径。
    参数:
        rel_path: 相对路径，如 "records/images/record_xxx.png"
    返回:
        绝对路径字符串
    """
    if not rel_path:
        return ""
    if os.path.isabs(rel_path):
        return rel_path
    return os.path.join(_project_root(), rel_path)


# ========== 推理功能：记录查询（返回三个值）==========
def _records_query(auth, q: str, page: int):
    """
    分页查询识别记录（供 Gradio 回调使用）
    参数:
        auth: 用户认证信息（保留，用于权限控制）
        q: 搜索关键字（可选）
        page: 页码（从1开始）
    返回:
        rows: 表格行数据（二维列表）
        info: 分页信息字符串
        raw_items: 原始数据列表（包含 id 等字段，用于选中行时取 id）
    """
    try:
        query = (q or "").strip()
        page_num = max(int(page or 1), 1)
        rows, total_pages, total = _get_db().list_records(query, page_num, per_page=10)
        items = [_row_to_dict(row) for row in rows]
        page_num = min(page_num, total_pages) if total_pages > 0 else page_num
        return _records_to_rows(items), f"第 {page_num} 页 / 共 {total_pages} 页，共 {total} 条记录", items
    except Exception as exc:
        return [], f"加载记录失败: {exc}", []


# ========== 推理功能：记录详情 ==========
def _record_detail(auth, record_id):
    """
    根据记录 ID 获取详情（含图片绝对路径）
    参数:
        auth: 用户认证信息
        record_id: 记录 ID
    返回:
        image_value: 图片绝对路径（或 None）
        detail: 详情字典（或 None）
        msg: 提示字符串
    """
    try:
        rid = int(record_id or 0)
    except (TypeError, ValueError):
        return None, None, "请输入有效的记录 ID"
    if rid <= 0:
        return None, None, "请输入有效的记录 ID"
    try:
        row = _get_db().get_record_by_id(rid)
        item = _row_to_dict(row)
        if not item:
            return None, None, f"未找到记录: {rid}"
        thumb_path = _to_absolute_project_path(str(item.get("thumb") or ""))
        image_value = thumb_path if thumb_path and os.path.exists(thumb_path) else None
        detail = {
            "id": item.get("id"),
            "label": item.get("label"),
            "confidence": item.get("confidence"),
            "elapsed": item.get("elapsed"),
            "model": item.get("model"),
            "weight": item.get("weight"),
            "user": item.get("user"),
            "thumb": item.get("thumb"),
            "created_at": item.get("created_at"),
        }
        return image_value, detail, "详情加载完成"
    except Exception as exc:
        return None, None, f"加载详情失败: {exc}"


# ========== 推理功能：导出 CSV ==========
def _records_export(auth, q: str):
    """
    导出识别记录为 CSV 文件（保存到系统临时目录）
    参数:
        auth: 用户认证信息
        q: 搜索关键字（可选）
    返回:
        export_path: 导出的文件路径（或 None）
        msg: 提示字符串
    """
    try:
        query = (q or "").strip()
        items = [_row_to_dict(row) for row in _get_db().export_records(query)]
        export_dir = os.path.join(tempfile.gettempdir(), "garbage_classifier_export")
        os.makedirs(export_dir, exist_ok=True)
        export_path = os.path.join(export_dir, f"records_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        with open(export_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "thumb", "label", "confidence", "elapsed", "model", "weight", "user", "created_at"])
            for item in items:
                writer.writerow([
                    item.get("id"),
                    item.get("thumb"),
                    item.get("label"),
                    item.get("confidence"),
                    item.get("elapsed"),
                    item.get("model"),
                    item.get("weight"),
                    item.get("user"),
                    item.get("created_at"),
                ])
        return export_path, f"导出完成, 共 {len(items)} 条记录"
    except Exception as exc:
        return None, f"导出失败: {exc}"


def _build_records_panel_state(auth, query: str = "", page: int = 1):
    rows, info, raw = _records_query(auth, query, page)
    # 返回表格行、分页信息、记录ID初始0、无图片、无详情JSON、详情提示空、原始数据列表
    return rows, info, 0, None, None, "", raw


# ========== 模型管理回调 ==========
def _models_set_current(auth, name):
    model_name = (name or "").strip()
    if not model_name:
        return "请选择要设为当前的模型"
    try:
        db = _get_db()
        if not db.check_model_exists(model_name):
            return f"模型不存在：{model_name}"
        db.set_setting(_CURRENT_MODEL_KEY, model_name)
        return f"当前模型已设置为：{model_name}"
    except Exception as exc:
        return f"设置当前模型失败：{exc}"


def _models_upload(auth, file_obj):
    """
    上传 .pth 模型文件并写入数据库
    参数:
        auth: 用户认证信息（保留用于权限校验）
        file_obj: Gradio File 组件返回的对象
    返回:
        提示消息字符串
    """
    if file_obj is None:
        return "请先选择 .pth 文件"
    src_path = getattr(file_obj, "name", None) or getattr(file_obj, "path", None) or ""
    src_path = str(src_path or "").strip()
    if not src_path or not os.path.exists(src_path):
        return "上传文件不存在或已失效，请重新选择"
    filename = os.path.basename(src_path)
    if not filename.lower().endswith(".pth"):
        return "只支持上传 .pth 模型文件"
    upload_dir = os.path.join(_project_root(), "models", "uploaded")
    os.makedirs(upload_dir, exist_ok=True)
    dest_path = os.path.join(upload_dir, filename)
    if os.path.abspath(src_path) != os.path.abspath(dest_path):
        shutil.copy2(src_path, dest_path)
    arch = "uploaded"
    try:
        checkpoint = torch.load(dest_path, map_location="cpu")
        arch = str(checkpoint.get("arch") or arch)
    except Exception:
        pass
    try:
        _register_model_record(dest_path, arch)
        return f"上传成功: {filename}"
    except Exception as exc:
        return f"上传成功，但写入数据库失败: {exc}"


def _profile_update(auth, name: str, gender: str, email: str, phone: str) -> str:
    """更新个人信息（占位实现）"""
    user_id = _auth_user_id(auth)
    if not user_id:
        return "请先登录"
    try:
        db = _get_db()
        return f"个人信息更新功能暂未实现 (name={name})"
    except Exception as exc:
        return f"更新失败: {exc}"


def _users_delete(auth, user_id):
    if not _is_admin(auth):
        return "无权限"
    return f"删除用户 {user_id} 功能暂未实现"


def _users_list(auth):
    if not _is_admin(auth):
        return [], "无权限"
    return [], "用户列表功能暂未实现"


def _users_update(auth, user_id, name, gender, email, phone, role):
    if not _is_admin(auth):
        return "无权限"
    return f"更新用户 {user_id} 功能暂未实现"


# ========== 训练调度函数 ==========
def _parse_training_progress(line: str, fallback_total: int) -> tuple[int, int]:
    match = _EPOCH_PROGRESS_RE.search(line or "")
    if not match:
        return 0, fallback_total
    current = match.group(1) or match.group(3) or "0"
    total = match.group(2) or match.group(4) or str(fallback_total)
    try:
        return max(int(current), 0), max(int(total), 1)
    except ValueError:
        return 0, fallback_total


def _sanitize_training_line(line: str) -> str:
    text = (line or "").replace("\r", "").rstrip()
    if not text:
        return ""
    text = text.replace("\ufffd", "")
    text = re.sub(r"\?{3,}", "", text)
    if "%" in text and "|" in text and "/" in text:
        text = re.sub(r"[^\x20-\x7E\u4e00-\u9fff]+", "", text)
        text = text.replace("|", "|")
    return text.strip()


def _train_model(auth, model_name, save_path, epochs, batch_size, lr, weight_decay):
    """Gradio 生成器：启动子进程训练并流式返回日志。"""
    model_name = (model_name or "").strip().lower()
    save_path = (save_path or "").strip() or _model_output_dir(model_name)
    total_epochs = max(int(epochs or 1), 1)
    script_map = {
        "resnet50": "train_resnet50.py",
        "alexnet": "train_alexnet.py",
        "cnn": "train_cnn.py",
    }
    script_name = script_map.get(model_name)
    if not script_name:
        yield "请选择要训练的模型。", "未开始", 0
        return
    train_dir = os.path.join(_project_root(), "train")
    script_path = os.path.join(train_dir, script_name)
    if not os.path.exists(script_path):
        yield f"未找到训练脚本: {script_path}", "训练脚本不存在", 0
        return
    os.makedirs(save_path, exist_ok=True)
    run_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    cmd = [
        sys.executable,
        "-u",
        script_path,
        "--out-dir",
        save_path,
        "--epochs",
        str(total_epochs),
        "--batch-size",
        str(int(batch_size or 8)),
        "--lr",
        str(float(lr or 1e-3)),
        "--weight-decay",
        str(float(weight_decay or 1e-4)),
        "--num-workers",
        str(_TRAIN_NUM_WORKERS),
        "--run-name",
        run_name,
    ]
    if _TRAIN_FORCE_CPU:
        cmd.append("--cpu")
    if model_name == "resnet50" and _TRAIN_RESNET50_PRETRAINED:
        cmd.append("--pretrained")
    yield "开始训练...\n" + " ".join(cmd), "正在启动训练进程", 0
    child_env = os.environ.copy()
    child_env["PYTHONIOENCODING"] = "utf-8"
    child_env["PYTHONUTF8"] = "1"
    child_env["TQDM_ASCII"] = "1"

    process = subprocess.Popen(
        cmd,
        cwd=_project_root(),
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding=locale.getpreferredencoding(False) or "utf-8",
        errors="replace",
        bufsize=1,
    )
    logs: list[str] = []
    progress_value = 0
    status_text = "训练中..."
    saved_checkpoint_path = ""
    assert process.stdout is not None
    for line in process.stdout:
        clean_line = _sanitize_training_line(line)
        if not clean_line:
            continue
        logs.append(clean_line)
        if clean_line.startswith("saved best checkpoint to:"):
            saved_checkpoint_path = clean_line.split(":", 1)[1].strip()
        current_epoch, parsed_total = _parse_training_progress(clean_line, total_epochs)
        if current_epoch > 0:
            progress_value = min(int(current_epoch * 100 / max(parsed_total, 1)), 100)
            status_text = f"训练中:第{current_epoch}/{parsed_total}轮"
        else:
            status_text = clean_line
        yield "\n".join(logs[-400:]), status_text, progress_value
    return_code = process.wait()
    if return_code == 0:
        if not saved_checkpoint_path:
            saved_checkpoint_path = os.path.join(save_path, f"{run_name}.pth")
        try:
            _register_model_record(saved_checkpoint_path, model_name)
            _get_db().set_setting(_CURRENT_MODEL_KEY, os.path.basename(saved_checkpoint_path))
        except Exception as exc:
            logs.append(f"[数据库写入失败]{exc}")
        logs.append("[训练完成]")
        status_text = "训练完成"
        progress_value = 100
    else:
        logs.append(f"[训练失败]退出码:{return_code}")
        status_text = "训练失败"
    yield "\n".join(logs[-400:]), status_text, progress_value


# ========== 登录相关辅助 ==========
def _is_hashed_password(stored_password: str) -> bool:
    return str(stored_password or "").startswith(("pbkdf2:", "scrypt:", "argon2:"))


def _verify_password(stored_password: str, provided_password: str) -> bool:
    stored = str(stored_password or "")
    provided = provided_password or ""
    if not stored or not provided or not _is_hashed_password(stored):
        return False
    try:
        return check_password_hash(stored, provided)
    except Exception:
        return False


def _hash_password(password: str) -> str:
    return generate_password_hash(password, method="pbkdf2:sha256")


def _is_admin(auth: Any) -> bool:
    return bool(auth and isinstance(auth, dict) and str(auth.get("role") or "").lower() == "admin")


def _auth_user_id(auth: Any) -> int | None:
    if not auth or not isinstance(auth, dict):
        return None
    try:
        user_id = int(auth.get("user_id") or 0)
    except (TypeError, ValueError):
        return None
    return user_id if user_id > 0 else None


def _login(username: str, password: str):
    username = (username or "").strip()
    password = password or ""
    if not username:
        return None, "请输入用户名"
    if not password:
        return None, "请输入密码"

    try:
        row = _get_db().get_user_by_username(username)
    except Exception as exc:
        return None, f"登录失败: {exc}"

    user = _row_to_dict(row)
    if not user:
        return None, "用户名或密码错误"
    if not _verify_password(str(user.get("password") or ""), password):
        return None, "用户名或密码错误"

    is_active = user.get("is_active", 1)
    if is_active in (0, False, "0"):
        return None, "账号已被禁用, 请联系管理员"

    display_name = (user.get("name") or user.get("username") or username).strip()
    return {
        "user_id": user.get("id"),
        "username": user.get("username") or username,
        "role": user.get("role") or "user",
        "name": display_name,
    }, f"欢迎, {display_name}"


def _logout():
    return None, "已退出登录"


def _users_create(auth, username, password, role, name, gender, email, phone):
    if not _is_admin(auth):
        return "无权限：仅管理员可创建用户"
    username = (username or "").strip()
    password = password or ""
    if not username:
        return "请输入用户名"
    if not password:
        return "请输入密码"
    role = (role or "user").strip() or "user"
    try:
        db = _get_db()
        if db.get_user_by_username(username):
            return f"用户名已存在: {username}"
        db.create_user(
            username,
            _hash_password(password),
            role,
            (name or "").strip(),
            (gender or "").strip(),
            (email or "").strip(),
            (phone or "").strip(),
        )
        return f"用户创建成功: {username}"
    except Exception as exc:
        return f"创建失败: {exc}"


def _users_reset_password(auth, user_id, new_password):
    if not _is_admin(auth):
        return "无权限：仅管理员可重置密码"
    try:
        uid = int(user_id or 0)
    except (TypeError, ValueError):
        return "请输入有效的用户 ID"
    if uid <= 0:
        return "请输入有效的用户 ID"
    new_password = new_password or ""
    if not new_password:
        return "请输入新密码"
    try:
        db = _get_db()
        if not _row_to_dict(db.get_user_by_id(uid)):
            return f"用户不存在: {uid}"
        db.update_user_password(uid, _hash_password(new_password))
        return f"用户 {uid} 密码已重置"
    except Exception as exc:
        return f"重置失败: {exc}"


def _profile_change_password(auth, old_pwd, new_pwd):
    user_id = _auth_user_id(auth)
    if not user_id:
        return "请先登录"
    old_pwd = old_pwd or ""
    new_pwd = (new_pwd or "").strip()
    if not old_pwd:
        return "请输入旧密码"
    if not new_pwd:
        return "请输入新密码"
    try:
        user = _row_to_dict(_get_db().get_user_by_id(user_id))
        if not user:
            return "用户不存在"
        if not _verify_password(str(user.get("password") or ""), old_pwd):
            return "旧密码错误"
        _get_db().update_user_password(user_id, _hash_password(new_pwd))
        return "密码修改成功"
    except Exception as exc:
        return f"密码修改失败: {exc}"


def _profile_get(auth):
    return {
        "name": "示例用户",
        "gender": "未知",
        "email": "demo@example.com",
        "phone": "13000000000",
    }, "个人信息加载完成"


def build_login_panel() -> dict[str, Any]:
    panel = gr.Column(visible=True, elem_id="login-panel")
    with panel:
        with gr.Column(elem_id="login-wrap"):
            gr.Markdown("## 登录")
            username = gr.Textbox(label="用户名")
            password = gr.Textbox(label="密码", type="password")
            login_btn = gr.Button("登录", variant="primary")
            status = gr.Textbox(label="状态", interactive=False)
    return {
        "panel": panel,
        "username": username,
        "password": password,
        "login_btn": login_btn,
        "status": status,
    }


def build_main_panel(auth_state, callbacks: dict[str, Any]) -> dict[str, Any]:
    panel = gr.Column(visible=False, elem_id="main-panel")

    with panel:
        gr.Markdown("## 欢迎使用垃圾分类识别系统")
        with gr.Tabs():
            with gr.Tab("个人信息"):
                logout_btn = gr.Button("退出登录")
                pf_name = gr.Textbox(label="姓名")
                pf_gender = gr.Textbox(label="性别")
                pf_email = gr.Textbox(label="邮箱")
                pf_phone = gr.Textbox(label="电话")

            with gr.Tab("图像识别"):
                rec_image = gr.Image(label="上传图片", type="pil")
                rec_weight = gr.Dropdown(
                    label="权重文件",
                    choices=callbacks["refresh_weights"](),
                    value=callbacks["get_current_weight"]() or None,
                )
                rec_refresh = gr.Button("刷新权重列表")
                rec_btn = gr.Button("开始识别")
                rec_msg = gr.Textbox(label="提示", interactive=False)
                rec_preview = gr.Image(label="预览", interactive=False)
                rec_result = gr.JSON(label="识别结果")
                rec_topk = gr.Dataframe(label="Top-K", headers=["label", "score"], interactive=False)

                def _weights_update():
                    choices = callbacks["refresh_weights"]()
                    cur = callbacks["get_current_weight"]() or (choices[0] if choices else None)
                    return gr.update(choices=choices, value=cur)

                rec_refresh.click(_weights_update, inputs=None, outputs=[rec_weight])
                rec_btn.click(
                    callbacks["recognize"],
                    inputs=[auth_state, rec_image, rec_weight],
                    outputs=[rec_preview, rec_result, rec_msg, rec_topk],
                )

            with gr.Tab("识别记录"):
                rec_q = gr.Textbox(label="搜索关键词", value="")
                rec_page = gr.Number(label="页码", value=1, precision=0, minimum=1)
                rec_query_btn = gr.Button("查询")
                rec_table = gr.Dataframe(
                    label="识别记录",
                    headers=["id", "label", "confidence", "model", "weight", "user", "created_at"],
                    interactive=False,
                    wrap=True,
                )
                rec_info = gr.Textbox(label="分页信息", interactive=False)
                rec_id = gr.Number(label="记录 ID", value=0, precision=0)
                rec_detail_btn = gr.Button("查看详情")
                rec_detail_image = gr.Image(label="快照预览", interactive=False)
                rec_detail_json = gr.JSON(label="详情 JSON")
                rec_detail_msg = gr.Textbox(label="详情提示", interactive=False)
                rec_export_btn = gr.Button("导出 CSV")
                rec_export_file = gr.File(label="下载导出文件", interactive=False)
                rec_table_data_state = gr.State([])  # 存储当前表格的原始数据，用于选中行时取 id

                # 辅助函数：查询时同时更新表格、分页信息、原始数据状态
                def _query_records(auth, q, page):
                    rows, info, raw = callbacks["records_query"](auth, q, page)
                    return rows, info, raw

                # 辅助函数：选中表格行时，从原始数据中提取 id 并填充到 rec_id
                def _pick_record_id(evt: gr.SelectData, table_data):
                    idx = evt.index[0]  # 行索引
                    if table_data and idx < len(table_data):
                        return table_data[idx].get("id")
                    return 0

                rec_query_btn.click(
                    _query_records,
                    inputs=[auth_state, rec_q, rec_page],
                    outputs=[rec_table, rec_info, rec_table_data_state],
                )
                rec_table.select(
                    _pick_record_id,
                    inputs=[rec_table_data_state],
                    outputs=[rec_id],
                )
                rec_detail_btn.click(
                    callbacks["record_detail"],
                    inputs=[auth_state, rec_id],
                    outputs=[rec_detail_image, rec_detail_json, rec_detail_msg],
                )
                rec_export_btn.click(
                    callbacks["records_export"],
                    inputs=[auth_state, rec_q],
                    outputs=[rec_export_file, rec_info],
                )

            with gr.Tab("模型管理（管理员）"):
                models_refresh = gr.Button("刷新模型列表")
                models_table = gr.Dataframe(
                    label="模型列表",
                    headers=["name", "size_mb", "num_classes", "epoch", "val_acc", "updated_at", "is_current"],
                    interactive=False,
                )
                current_model = gr.Textbox(label="当前模型", interactive=False)
                models_msg = gr.Textbox(label="提示", interactive=False)
                upload_file = gr.File(label="上传 .pth 权重")
                upload_btn = gr.Button("上传")
                upload_msg = gr.Textbox(label="上传提示", interactive=False)
                set_current_dd = gr.Dropdown(
                    label="设置当前模型",
                    choices=callbacks["refresh_weights"](),
                )
                set_current_btn = gr.Button("设置")
                set_current_msg = gr.Textbox(label="设置提示", interactive=False)
                gr.Markdown("## 模型训练")
                with gr.Accordion("训练模型面板", open=True):
                    train_model_name = gr.Dropdown(
                        label="选择模型",
                        choices=["resnet50", "alexnet", "cnn"],
                        value="resnet50",
                    )
                    train_save_path = gr.Textbox(
                        label="模型保存目录",
                        value=_model_output_dir("resnet50"),
                        interactive=True,
                    )
                    with gr.Row():
                        train_epochs = gr.Number(label="训练轮数", value=2, precision=0)
                        train_batch_size = gr.Number(label="批大小", value=16, precision=0)
                    with gr.Row():
                        train_lr = gr.Number(label="学习率", value=0.001, precision=6)
                        train_weight_decay = gr.Number(label="权重衰减", value=0.0001, precision=6)
                    train_btn = gr.Button("开始训练", variant="primary")
                    train_status = gr.Textbox(label="训练状态", value="未开始", interactive=False)
                    train_progress = gr.Slider(
                        label="训练进度", minimum=0, maximum=100, step=1, value=0, interactive=False
                    )
                    train_log = gr.Textbox(label="训练日志", lines=18, max_lines=24, interactive=False)

        # 模型管理内部回调函数
        def _models_refresh(auth):
            items, cur, msg = callbacks["models_list"](auth)
            choices = callbacks["refresh_weights"]()
            return _model_panel_state(items, cur, msg, choices)

        def _update_train_save_path(model_name):
            model_name = (model_name or "resnet50").strip().lower()
            return _model_output_dir(model_name)

        models_refresh.click(
            _models_refresh,
            inputs=[auth_state],
            outputs=[models_table, current_model, models_msg, set_current_dd],
        )
        upload_btn.click(
            callbacks["models_upload"],
            inputs=[auth_state, upload_file],
            outputs=[upload_msg],
        ).then(
            _models_refresh,
            inputs=[auth_state],
            outputs=[models_table, current_model, models_msg, set_current_dd],
        )
        set_current_btn.click(
            callbacks["models_set_current"],
            inputs=[auth_state, set_current_dd],
            outputs=[set_current_msg],
        ).then(
            _models_refresh,
            inputs=[auth_state],
            outputs=[models_table, current_model, models_msg, set_current_dd],
        )
        train_model_name.change(
            _update_train_save_path,
            inputs=[train_model_name],
            outputs=[train_save_path],
        )
        train_btn.click(
            callbacks["train_model"],
            inputs=[
                auth_state,
                train_model_name,
                train_save_path,
                train_epochs,
                train_batch_size,
                train_lr,
                train_weight_decay,
            ],
            outputs=[train_log, train_status, train_progress],
        ).then(
            _models_refresh,
            inputs=[auth_state],
            outputs=[models_table, current_model, models_msg, set_current_dd],
        )

    return {
        "panel": panel,
        "logout_btn": logout_btn,
        "profile_fill_outputs": [pf_name, pf_gender, pf_email, pf_phone],
        "models_table": models_table,
        "current_model": current_model,
        "models_msg": models_msg,
        "set_current_dd": set_current_dd,
        "set_current_btn": set_current_btn,
        "set_current_msg": set_current_msg,
        "train_model_name": train_model_name,
        "train_save_path": train_save_path,
        "train_epochs": train_epochs,
        "train_batch_size": train_batch_size,
        "train_lr": train_lr,
        "train_weight_decay": train_weight_decay,
        "train_btn": train_btn,
        "train_status": train_status,
        "train_progress": train_progress,
        "train_log": train_log,
        "models_outputs": [models_table, current_model, models_msg, set_current_dd],
        "records_outputs": [rec_table, rec_info, rec_id, rec_detail_image, rec_detail_json, rec_detail_msg, rec_table_data_state],
    }


def build_app() -> gr.Blocks:
    with gr.Blocks(title="垃圾分类识别系统", css=APP_CSS) as demo:
        auth_state = gr.State(None)
        gr.Markdown("# 垃圾分类识别系统（教学页面）")

        def _apply_auth_ui(auth, status: str):
            logged_in = bool(auth and isinstance(auth, dict) and auth.get("user_id"))
            return (
                gr.update(visible=not logged_in),
                gr.update(visible=logged_in),
                status if logged_in else ((status or "").strip() or "请登录后继续"),
            )

        def _pf_fill(auth):
            if not auth:
                return "", "", "", ""
            data, _ = _profile_get(auth)
            return data.get("name", ""), data.get("gender", ""), data.get("email", ""), data.get("phone", "")

        def _load_models_ui(auth):
            if not auth:
                return [], "", "请先登录", gr.update()
            try:
                items, cur, msg = _models_list(auth)
                return _model_panel_state(items, cur, msg, _refresh_weights())
            except Exception as exc:
                return [], "", f"加载模型列表失败: {exc}", gr.update()

        login_ui = build_login_panel()
        main_ui = build_main_panel(
            auth_state,
            {
                "get_current_weight": _get_current_weight,
                "models_list": _models_list,
                "models_set_current": _models_set_current,
                "train_model": _train_model,
                "models_upload": _models_upload,
                "profile_change_password": _profile_change_password,
                "profile_fill": _pf_fill,
                "profile_update": _profile_update,
                "record_detail": _record_detail,
                "records_export": _records_export,
                "records_query": _records_query,
                "recognize": _recognize,
                "refresh_weights": _refresh_weights,
                "users_create": _users_create,
                "users_delete": _users_delete,
                "users_list": _users_list,
                "users_reset_password": _users_reset_password,
                "users_update": _users_update,
            }
        )

        demo.load(
            lambda: gr.update(visible=True),
            inputs=None,
            outputs=[main_ui["panel"]],
        ).then(
            lambda: gr.update(visible=False),
            inputs=None,
            outputs=[main_ui["panel"]],
        )

        login_ui["login_btn"].click(
            _login,
            inputs=[login_ui["username"], login_ui["password"]],
            outputs=[auth_state, login_ui["status"]],
        ).then(
            _apply_auth_ui,
            inputs=[auth_state, login_ui["status"]],
            outputs=[login_ui["panel"], main_ui["panel"], login_ui["status"]],
        ).then(
            _pf_fill,
            inputs=[auth_state],
            outputs=main_ui["profile_fill_outputs"],
        ).then(
            _load_models_ui,
            inputs=[auth_state],
            outputs=main_ui["models_outputs"],
        ).then(
            _build_records_panel_state,
            inputs=[auth_state],
            outputs=main_ui["records_outputs"],
        )

        main_ui["logout_btn"].click(
            _logout,
            inputs=None,
            outputs=[auth_state, login_ui["status"]],
        ).then(
            _apply_auth_ui,
            inputs=[auth_state, login_ui["status"]],
            outputs=[login_ui["panel"], main_ui["panel"], login_ui["status"]],
        ).then(
            lambda: ("", "", "", ""),
            inputs=None,
            outputs=main_ui["profile_fill_outputs"],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    # 将 css 移到 launch 中可以消除警告，但暂时保持兼容
    app.launch(server_name="127.0.0.1", server_port=int(os.environ.get("PORT", _CFG.port)))