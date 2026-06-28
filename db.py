"""数据库操作工具类 - 封装所有数据库相关操作"""
import os
import re
import sqlite3
from urllib.parse import urlparse, unquote
from contextlib import contextmanager
from typing import Optional, List, Dict, Any, Tuple


class Database:
    """数据库操作工具类（SQLite / MySQL）"""

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._backend = self._detect_backend(db_path)

    @staticmethod
    def _detect_backend(db_path: str) -> str:
        v = (db_path or "").strip().lower()
        if v.startswith("mysql://") or v.startswith("mysql+pymysql://"):
            return "mysql"
        raise ValueError("当前项目仅支持 MySQL：请使用 DB_URL 形式 mysql://... 或 mysql+pymysql://...")

    def _parse_mysql_url(self) -> Dict[str, Any]:
        url = (self._db_path or "").strip()
        if url.startswith("mysql+pymysql://"):
            url = "mysql://" + url[len("mysql+pymysql://") :]
        p = urlparse(url)
        if p.scheme != "mysql":
            raise ValueError("Invalid MySQL DB URL")

        db_name = (p.path or "").lstrip("/")
        if not db_name:
            raise ValueError("MySQL DB URL missing database name")

        return {
            "host": p.hostname or "127.0.0.1",
            "port": int(p.port or 3306),
            "user": unquote(p.username or "root"),
            "password": unquote(p.password or ""),
            "database": db_name,
            "charset": "utf8mb4",
        }

    def _adapt_placeholders(self, sql: str) -> str:
        if self._backend == "sqlite":
            return sql
        return re.sub(r"\?", "%s", sql)

    def _row_get(self, row: Any, key: str, default: Any = None) -> Any:
        try:
            if row is None:
                return default
            if isinstance(row, dict):
                return row.get(key, default)
            return row[key]
        except Exception:
            return default

    @contextmanager
    def _conn(self):
        """数据库连接上下文管理器"""
        if self._backend == "sqlite":
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
            finally:
                conn.close()
            return

        import pymysql

        cfg = self._parse_mysql_url()
        conn = pymysql.connect(
            host=cfg["host"],
            port=cfg["port"],
            user=cfg["user"],
            password=cfg["password"],
            database=cfg["database"],
            charset=cfg["charset"],
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=False,
        )
        try:
            yield conn
        finally:
            conn.close()

    def init_tables(self) -> None:
        """初始化数据库表结构"""
        with self._conn() as conn:
            if self._backend == "sqlite":
                # 用户表
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT NOT NULL UNIQUE,
                        password TEXT NOT NULL,
                        name TEXT,
                        gender TEXT,
                        email TEXT,
                        phone TEXT,
                        role TEXT
                    )
                    """
                )
                try:
                    conn.execute("ALTER TABLE users ADD COLUMN is_active INTEGER DEFAULT 1")
                except sqlite3.OperationalError:
                    pass

                # 识别记录表
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS records (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        thumb TEXT,
                        label TEXT,
                        confidence REAL,
                        elapsed REAL,
                        model TEXT,
                        weight TEXT,
                        user TEXT,
                        created_at TEXT
                    )
                    """
                )

                # 设置表
                conn.execute("CREATE TABLE IF NOT EXISTS settings (k TEXT PRIMARY KEY, v TEXT)")

                # 模型表
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS models (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL UNIQUE,
                        path TEXT NOT NULL,
                        arch TEXT,
                        size_mb REAL,
                        num_classes INTEGER,
                        epoch INTEGER,
                        val_acc REAL,
                        created_at TEXT,
                        updated_at TEXT
                    )
                    """
                )
                try:
                    conn.execute("ALTER TABLE models ADD COLUMN arch TEXT")
                except sqlite3.OperationalError:
                    pass

                conn.commit()
                return

            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS users (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        username VARCHAR(255) NOT NULL UNIQUE,
                        password TEXT NOT NULL,
                        name VARCHAR(255) NULL,
                        gender VARCHAR(64) NULL,
                        email VARCHAR(255) NULL,
                        phone VARCHAR(64) NULL,
                        role VARCHAR(64) NULL,
                        is_active TINYINT NOT NULL DEFAULT 1
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                    """
                )

                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS records (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        thumb TEXT,
                        label VARCHAR(255),
                        confidence DOUBLE,
                        elapsed DOUBLE,
                        model VARCHAR(255),
                        weight VARCHAR(255),
                        user VARCHAR(255),
                        created_at VARCHAR(64)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                    """
                )

                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS settings (
                        k VARCHAR(255) PRIMARY KEY,
                        v TEXT
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                    """
                )

                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS models (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        name VARCHAR(255) NOT NULL UNIQUE,
                        path TEXT NOT NULL,
                        arch VARCHAR(255) NULL,
                        size_mb DOUBLE,
                        num_classes INT,
                        epoch INT,
                        val_acc DOUBLE,
                        created_at VARCHAR(64),
                        updated_at VARCHAR(64)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                    """
                )

            conn.commit()

    # ========== 设置操作 ==========

    def get_setting(self, key: str, default: str = "") -> str:
        """获取设置值"""
        with self._conn() as conn:
            sql = self._adapt_placeholders("SELECT v FROM settings WHERE k = ?")
            if self._backend == "sqlite":
                row = conn.execute(sql, (key,)).fetchone()
            else:
                with conn.cursor() as cur:
                    cur.execute(sql, (key,))
                    row = cur.fetchone()
            v = self._row_get(row, "v")
            return str(v) if v is not None else default

    def set_setting(self, key: str, value: str) -> None:
        """设置值"""
        with self._conn() as conn:
            if self._backend == "sqlite":
                conn.execute(
                    "INSERT INTO settings (k, v) VALUES (?, ?) ON CONFLICT(k) DO UPDATE SET v = excluded.v",
                    (key, value),
                )
                conn.commit()
                return

            sql = "INSERT INTO settings (k, v) VALUES (%s, %s) ON DUPLICATE KEY UPDATE v = VALUES(v)"
            with conn.cursor() as cur:
                cur.execute(sql, (key, value))
            conn.commit()

    # ========== 用户操作 ==========

    def get_user_by_id(self, user_id: int) -> Optional[Any]:
        """根据ID获取用户"""
        with self._conn() as conn:
            sql = self._adapt_placeholders(
                "SELECT id, username, password, name, gender, email, phone, role, COALESCE(is_active, 1) AS is_active FROM users WHERE id = ?"
            )
            if self._backend == "sqlite":
                return conn.execute(sql, (user_id,)).fetchone()
            with conn.cursor() as cur:
                cur.execute(sql, (user_id,))
                return cur.fetchone()

    def get_user_by_username(self, username: str) -> Optional[Any]:
        """根据用户名获取用户"""
        with self._conn() as conn:
            sql = self._adapt_placeholders(
                "SELECT id, username, password, name, gender, email, phone, role, COALESCE(is_active, 1) AS is_active FROM users WHERE username = ?"
            )
            if self._backend == "sqlite":
                return conn.execute(sql, (username,)).fetchone()
            with conn.cursor() as cur:
                cur.execute(sql, (username,))
                return cur.fetchone()

    def list_users(self, search: str = "") -> List[Any]:
        """获取用户列表，支持搜索"""
        with self._conn() as conn:
            if search:
                like = f"%{search}%"
                sql = self._adapt_placeholders(
                    "SELECT id, username, name, gender, email, phone, role, COALESCE(is_active, 1) AS is_active FROM users WHERE username LIKE ? OR name LIKE ? OR email LIKE ? OR phone LIKE ? ORDER BY id ASC"
                )
                params = (like, like, like, like)
                if self._backend == "sqlite":
                    return conn.execute(sql, params).fetchall()
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    return cur.fetchall()
            else:
                sql = "SELECT id, username, name, gender, email, phone, role, COALESCE(is_active, 1) AS is_active FROM users ORDER BY id ASC"
                if self._backend == "sqlite":
                    return conn.execute(sql).fetchall()
                with conn.cursor() as cur:
                    cur.execute(sql)
                    return cur.fetchall()

    def create_user(self, username: str, password: str, role: str, name: str = "", gender: str = "", email: str = "", phone: str = "") -> None:
        """创建用户（password 须为已哈希后的密文）"""
        with self._conn() as conn:
            sql = self._adapt_placeholders(
                "INSERT INTO users (username, password, name, gender, email, phone, role, is_active) VALUES (?, ?, ?, ?, ?, ?, ?, 1)"
            )
            params = (username, password, name, gender, email, phone, role)
            if self._backend == "sqlite":
                conn.execute(sql, params)
            else:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
            conn.commit()

    def update_user(self, user_id: int, name: str, gender: str, email: str, phone: str, role: str, is_active: int) -> None:
        """更新用户信息"""
        with self._conn() as conn:
            sql = self._adapt_placeholders("UPDATE users SET name=?, gender=?, email=?, phone=?, role=?, is_active=? WHERE id=?")
            params = (name, gender, email, phone, role, is_active, user_id)
            if self._backend == "sqlite":
                conn.execute(sql, params)
            else:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
            conn.commit()

    def update_user_password(self, user_id: int, password: str) -> None:
        """更新用户密码（password 须为已哈希后的密文）"""
        with self._conn() as conn:
            sql = self._adapt_placeholders("UPDATE users SET password=? WHERE id=?")
            params = (password, user_id)
            if self._backend == "sqlite":
                conn.execute(sql, params)
            else:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
            conn.commit()

    def delete_user(self, user_id: int) -> None:
        """删除用户"""
        with self._conn() as conn:
            sql = self._adapt_placeholders("DELETE FROM users WHERE id=?")
            if self._backend == "sqlite":
                conn.execute(sql, (user_id,))
            else:
                with conn.cursor() as cur:
                    cur.execute(sql, (user_id,))
            conn.commit()

    def count_users(self) -> int:
        """获取用户总数"""
        with self._conn() as conn:
            sql = "SELECT COUNT(1) AS c FROM users"
            if self._backend == "sqlite":
                row = conn.execute(sql).fetchone()
            else:
                with conn.cursor() as cur:
                    cur.execute(sql)
                    row = cur.fetchone()
            return int(self._row_get(row, "c", 0) or 0)

    # ========== 记录操作 ==========

    def create_record(self, thumb: str, label: str, confidence: float, elapsed: float, model: str, weight: str, user: str, created_at: str) -> None:
        """创建识别记录"""
        with self._conn() as conn:
            sql = self._adapt_placeholders(
                "INSERT INTO records (thumb, label, confidence, elapsed, model, weight, user, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
            )
            params = (thumb, label, confidence, elapsed, model, weight, user, created_at)
            if self._backend == "sqlite":
                conn.execute(sql, params)
            else:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
            conn.commit()

    def list_records(self, search: str = "", page: int = 1, per_page: int = 10) -> Tuple[List[sqlite3.Row], int, int]:
        """获取记录列表，返回 (记录列表, 总页数, 总数)"""
        with self._conn() as conn:
            if search:
                like = f"%{search}%"
                where = "WHERE label LIKE ? OR model LIKE ? OR weight LIKE ? OR user LIKE ? OR created_at LIKE ?"
                params = [like, like, like, like, like]
            else:
                where = ""
                params = []

            count_sql = self._adapt_placeholders(f"SELECT COUNT(1) AS c FROM records {where}")
            if self._backend == "sqlite":
                total_row = conn.execute(count_sql, params).fetchone()
            else:
                with conn.cursor() as cur:
                    cur.execute(count_sql, params)
                    total_row = cur.fetchone()
            total = int(self._row_get(total_row, "c", 0) or 0)
            total_pages = max(1, (total + per_page - 1) // per_page)
            page = min(page, total_pages)

            offset = (page - 1) * per_page
            list_sql = self._adapt_placeholders(
                f"SELECT id, thumb, label, confidence, elapsed, model, weight, user, created_at FROM records {where} ORDER BY id DESC LIMIT ? OFFSET ?"
            )
            list_params = params + [per_page, offset]
            if self._backend == "sqlite":
                rows = conn.execute(list_sql, list_params).fetchall()
            else:
                with conn.cursor() as cur:
                    cur.execute(list_sql, list_params)
                    rows = cur.fetchall()

            return rows, total_pages, total

    def export_records(self, search: str = "") -> List[sqlite3.Row]:
        """导出记录（不分页）"""
        with self._conn() as conn:
            if search:
                like = f"%{search}%"
                sql = self._adapt_placeholders(
                    "SELECT id, thumb, label, confidence, elapsed, model, weight, user, created_at FROM records WHERE label LIKE ? OR model LIKE ? OR weight LIKE ? OR user LIKE ? OR created_at LIKE ? ORDER BY id DESC"
                )
                params = [like, like, like, like, like]
                if self._backend == "sqlite":
                    return conn.execute(sql, params).fetchall()
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    return cur.fetchall()
            else:
                sql = "SELECT id, thumb, label, confidence, elapsed, model, weight, user, created_at FROM records ORDER BY id DESC"
                if self._backend == "sqlite":
                    return conn.execute(sql).fetchall()
                with conn.cursor() as cur:
                    cur.execute(sql)
                    return cur.fetchall()

    def get_record_by_id(self, record_id: int) -> Optional[sqlite3.Row]:
        """根据ID获取记录"""
        with self._conn() as conn:
            sql = self._adapt_placeholders(
                "SELECT id, thumb, label, confidence, elapsed, model, weight, user, created_at FROM records WHERE id = ?"
            )
            if self._backend == "sqlite":
                return conn.execute(sql, (record_id,)).fetchone()
            with conn.cursor() as cur:
                cur.execute(sql, (record_id,))
                return cur.fetchone()

    def count_records(self) -> int:
        """获取记录总数"""
        with self._conn() as conn:
            sql = "SELECT COUNT(1) AS c FROM records"
            if self._backend == "sqlite":
                row = conn.execute(sql).fetchone()
            else:
                with conn.cursor() as cur:
                    cur.execute(sql)
                    row = cur.fetchone()
            return int(self._row_get(row, "c", 0) or 0)

    # ========== 模型操作 ==========

    def list_model_names(self) -> set:
        """获取所有模型名称集合"""
        with self._conn() as conn:
            sql = "SELECT name FROM models"
            if self._backend == "sqlite":
                rows = conn.execute(sql).fetchall()
            else:
                with conn.cursor() as cur:
                    cur.execute(sql)
                    rows = cur.fetchall()
            return {self._row_get(r, "name") for r in rows}

    def create_model(self, name: str, path: str, arch: str, size_mb: float, num_classes: Optional[int], epoch: Optional[int], val_acc: Optional[float], created_at: str, updated_at: str) -> None:
        """创建模型记录"""
        with self._conn() as conn:
            sql = self._adapt_placeholders(
                "INSERT INTO models (name, path, arch, size_mb, num_classes, epoch, val_acc, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
            )
            params = (name, path, arch, size_mb, num_classes, epoch, val_acc, created_at, updated_at)
            if self._backend == "sqlite":
                conn.execute(sql, params)
            else:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
            conn.commit()

    def update_model(self, name: str, path: str, arch: str, size_mb: float, num_classes: Optional[int], epoch: Optional[int], val_acc: Optional[float], updated_at: str) -> None:
        """更新模型记录"""
        with self._conn() as conn:
            sql = self._adapt_placeholders(
                "UPDATE models SET path=?, arch=COALESCE(NULLIF(?, ''), arch), size_mb=?, num_classes=?, epoch=?, val_acc=?, updated_at=? WHERE name=?"
            )
            params = (path, arch, size_mb, num_classes, epoch, val_acc, updated_at, name)
            if self._backend == "sqlite":
                conn.execute(sql, params)
            else:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
            conn.commit()

    def delete_model_by_name(self, name: str) -> None:
        """根据名称删除模型"""
        with self._conn() as conn:
            sql = self._adapt_placeholders("DELETE FROM models WHERE name = ?")
            if self._backend == "sqlite":
                conn.execute(sql, (name,))
            else:
                with conn.cursor() as cur:
                    cur.execute(sql, (name,))
            conn.commit()

    def get_model_arch(self, name: str) -> str:
        """获取模型架构"""
        with self._conn() as conn:
            sql = self._adapt_placeholders("SELECT arch FROM models WHERE name = ?")
            if self._backend == "sqlite":
                row = conn.execute(sql, (name,)).fetchone()
            else:
                with conn.cursor() as cur:
                    cur.execute(sql, (name,))
                    row = cur.fetchone()
            return (self._row_get(row, "arch") or "")

    def list_model_arches(self) -> List[str]:
        """获取所有不同的模型架构列表"""
        with self._conn() as conn:
            sql = "SELECT DISTINCT arch FROM models WHERE arch IS NOT NULL AND TRIM(arch) <> '' ORDER BY arch ASC"
            if self._backend == "sqlite":
                rows = conn.execute(sql).fetchall()
            else:
                with conn.cursor() as cur:
                    cur.execute(sql)
                    rows = cur.fetchall()
            return [self._row_get(r, "arch") for r in rows]

    def list_models(self) -> List[Any]:
        """获取模型列表"""
        with self._conn() as conn:
            sql = "SELECT name, size_mb, num_classes, epoch, val_acc, updated_at FROM models ORDER BY updated_at DESC"
            if self._backend == "sqlite":
                return conn.execute(sql).fetchall()
            with conn.cursor() as cur:
                cur.execute(sql)
                return cur.fetchall()

    def check_model_exists(self, name: str) -> bool:
        """检查模型是否存在"""
        with self._conn() as conn:
            sql = self._adapt_placeholders("SELECT 1 AS v FROM models WHERE name = ?")
            if self._backend == "sqlite":
                row = conn.execute(sql, (name,)).fetchone()
            else:
                with conn.cursor() as cur:
                    cur.execute(sql, (name,))
                    row = cur.fetchone()
            return row is not None
