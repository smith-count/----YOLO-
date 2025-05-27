import tkinter as tk
from tkinter import messagebox
from Model.UI.Exe.model_page import YOLOAnalyzerApp
import mysql.connector
from mysql.connector import Error
import hashlib
from ttkbootstrap import Style

# 界面优化
style = Style(theme='sandstone')

"""数据库管理类"""
class DatabaseManager:

    def __init__(self):
        self.connection = None
        self.connect()

    """连接数据库"""
    def connect(self):
        try:
            self.connection = mysql.connector.connect(
                host='localhost',
                user='root',  # 替换为你的MySQL用户名
                password='123456',  # 替换为你的MySQL密码
                database='user_management'
            )
            if self.connection.is_connected():
                self.create_table()
        except Error as e:
            messagebox.showerror("数据库错误", f"无法连接数据库: {e}")

    """创建用户表(如果不存在)"""
    def create_table(self):
        try:
            cursor = self.connection.cursor() #返回控制游标
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    password VARCHAR(255) NOT NULL,
                    email VARCHAR(100),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.connection.commit()
        except Error as e:
            messagebox.showerror("数据库错误", f"创建表失败: {e}")

    def register_user(self, username, password, email):
        """注册新用户
        要求：
        - 用户名只能是数字
        - 密码必须是数字、字母和符号的组合
        """
        try:
            # 验证用户名是否为纯数字
            if not username.isdigit():
                return False, "用户名必须为纯数字"

            # 验证密码复杂度
            has_digit = any(c.isdigit() for c in password)
            has_letter = any(c.isalpha() for c in password)
            has_symbol = any(not c.isalnum() for c in password)

            if not (has_digit and has_letter and has_symbol):
                return False, "密码必须包含数字、字母和符号"

            # 检查密码长度（可选）
            if len(password) < 8:
                return False, "密码长度至少8位"

            cursor = self.connection.cursor()

            # 检查用户名是否已存在
            cursor.execute("SELECT username FROM users WHERE username = %s", username)
            if cursor.fetchone():
                return False, "用户名已存在"

            # 哈希密码
            hashed_password = hashlib.sha256(password.encode()).hexdigest()

            # 插入新用户
            cursor.execute("""
                INSERT INTO users (username, password, email)
                VALUES (%s, %s, %s)
            """, (username, hashed_password, email))
            self.connection.commit()
            return True, "注册成功"
        except Error as e:
            return False, f"数据库错误: {e}"
        finally:
            if 'cursor' in locals():
                cursor.close()

    """用户登录验证"""
    def login_user(self, username, password):
        try:
            cursor = self.connection.cursor(dictionary=True)
            # 获取用户信息
            cursor.execute("""
                SELECT id, username, password 
                FROM users 
                WHERE username = %s
            """, (username,))
            user = cursor.fetchone()

            if not user:
                return False, "用户名不存在"

            # 验证密码
            hashed_password = hashlib.sha256(password.encode()).hexdigest()
            if user['password'] == hashed_password:
                return True, "登录成功"
            else:
                return False, "密码错误"
        except Error as e:
            return False, f"数据库错误: {e}"

    """关闭数据库连接"""
    def close(self):
        if self.connection and self.connection.is_connected():
            self.connection.close()


class LoginApp:
    """登录应用程序GUI"""

    def __init__(self, root):
        self.root = root
        self.root.title("用户登录系统")
        self.root.geometry("300x200")
        self.db = DatabaseManager()

        # 创建主界面
        self.create_widgets()

        # 注册关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        """创建界面组件"""
        # 主框架
        self.main_frame = tk.Frame(self.root, padx=20, pady=20)
        self.main_frame.pack(expand=True, fill=tk.BOTH)

        # 登录界面
        self.create_login_ui()

        # 默认显示登录界面
        self.show_login()

    def create_login_ui(self):
        """创建登录界面组件"""
        self.login_frame = tk.Frame(self.main_frame)

        # 用户名
        tk.Label(self.login_frame, text="用户名:").grid(row=0, column=0, sticky=tk.E, pady=5)
        self.username_entry = tk.Entry(self.login_frame, width=25)
        self.username_entry.grid(row=0, column=1, pady=5)

        # 密码
        tk.Label(self.login_frame, text="密码:").grid(row=1, column=0, sticky=tk.E, pady=5)
        self.password_entry = tk.Entry(self.login_frame, width=25, show="*")
        self.password_entry.grid(row=1, column=1, pady=5)

        # 按钮
        button_frame = tk.Frame(self.login_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=10)

        tk.Button(button_frame, text="登录", command=self.login).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="注册", command=self.show_register).pack(side=tk.LEFT, padx=5)

    def create_register_ui(self):
        """创建注册界面组件
        注册要求：
        - 用户名必须是纯数字
        - 密码必须包含数字、字母和特殊符号的组合
        - 密码长度至少8位
        """
        self.register_frame = tk.Frame(self.main_frame)

        # 调整界面大小
        self.root.geometry("450x270")
        # 用户名行
        username_frame = tk.Frame(self.register_frame)
        username_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=5)

        tk.Label(username_frame, text="用户名:",width=10).pack(side=tk.LEFT)
        self.reg_username_entry = tk.Entry(username_frame, width=25)
        self.reg_username_entry.pack(side=tk.LEFT, padx=5)

        # 用户名要求提示
        username_hint = tk.Label(username_frame, text="(必须为纯数字)", fg="gray")
        username_hint.pack(side=tk.LEFT)

        # 密码行
        password_frame = tk.Frame(self.register_frame)
        password_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=5)

        tk.Label(password_frame, text="密码:",width=10).pack(side=tk.LEFT)
        self.reg_password_entry = tk.Entry(password_frame, width=25, show="*")
        self.reg_password_entry.pack(side=tk.LEFT, padx=5)

        # 密码要求提示
        password_hint = tk.Label(password_frame, text="(需包含数字、字母和符号)", fg="gray")
        password_hint.pack(side=tk.LEFT)

        # 确认密码行
        confirm_frame = tk.Frame(self.register_frame)
        confirm_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=5)

        tk.Label(confirm_frame, text="确认密码:",width=10).pack(side=tk.LEFT)
        self.reg_confirm_entry = tk.Entry(confirm_frame, width=25, show="*")
        self.reg_confirm_entry.pack(side=tk.LEFT, padx=5)

        # 邮箱行
        email_frame = tk.Frame(self.register_frame)
        email_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=5)

        tk.Label(email_frame, text="邮箱:",width=10).pack(side=tk.LEFT)
        self.reg_email_entry = tk.Entry(email_frame, width=25)
        self.reg_email_entry.pack(side=tk.LEFT, padx=5)

        # 密码强度实时检查
        self.password_strength_label = tk.Label(self.register_frame, text="", fg="red")
        self.password_strength_label.grid(row=4, column=0, columnspan=2)

        self.reg_password_entry.bind("<KeyRelease>", self.check_password_strength)

        # 按钮区域
        button_frame = tk.Frame(self.register_frame)
        button_frame.grid(row=5, column=0, columnspan=2, pady=10)

        tk.Button(button_frame, text="注册", command=self.register).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="返回", command=self.show_login).pack(side=tk.LEFT, padx=5)

    def check_password_strength(self, event):
        """实时检查密码强度"""
        password = self.reg_password_entry.get()

        has_digit = any(c.isdigit() for c in password)
        has_letter = any(c.isalpha() for c in password)
        has_symbol = any(not c.isalnum() for c in password)

        strength = []
        if not password:
            self.password_strength_label.config(text="")
            return

        if len(password) < 8:
            strength.append("长度至少8位")
        if not has_digit:
            strength.append("需要数字")
        if not has_letter:
            strength.append("需要字母")
        if not has_symbol:
            strength.append("需要特殊符号")

        if strength:
            self.password_strength_label.config(text="密码不符合要求: " + ", ".join(strength), fg="red")
        else:
            self.password_strength_label.config(text="密码强度: 强", fg="green")

    def show_login(self):
        """显示登录界面"""
        self.clear_frame()
        self.login_frame.pack(expand=True)

    def show_register(self):
        """显示注册界面"""
        if not hasattr(self, 'register_frame'):
            self.create_register_ui()
        self.clear_frame()
        self.register_frame.pack(expand=True)

    def clear_frame(self):
        """清除当前框架内容"""
        for widget in self.main_frame.winfo_children():
            widget.pack_forget()

    def login(self):
        """处理登录"""
        username = self.username_entry.get()
        password = self.password_entry.get()

        if not username or not password:
            messagebox.showwarning("输入错误", "请输入用户名和密码")
            return

        success, message = self.db.login_user(username, password)
        if success:
            messagebox.showinfo("登录成功", f"欢迎, {username}!")
            # 这里可以跳转到主界面
            # 销毁当前所有组件（如登录界面）
            for widget in self.root.winfo_children():
                widget.destroy()  # 彻底清除旧界面

            # 跳转到主界面
            main_app = YOLOAnalyzerApp(self.root)  # 将 root 传递给新类
            self.root.geometry("750x400")
            self.root.protocol("WM_DELETE_WINDOW", main_app.on_closing)

        else:
            messagebox.showerror("登录失败", message)

    def register(self):
        """处理注册"""
        username = self.reg_username_entry.get()
        password = self.reg_password_entry.get()
        confirm = self.reg_confirm_entry.get()
        email = self.reg_email_entry.get()

        # 验证输入
        if not username or not password:
            messagebox.showwarning("输入错误", "用户名和密码不能为空")
            return

        if password != confirm:
            messagebox.showwarning("输入错误", "两次输入的密码不一致")
            return

        # 注册用户
        success, message = self.db.register_user(username, password, email)
        if success:
            messagebox.showinfo("注册成功", message)
            self.show_login()
        else:
            messagebox.showerror("注册失败", message)

    def on_closing(self):
        """关闭窗口时的清理工作"""
        self.db.close()
        self.root.destroy()

if __name__ == "__main__":
    if not hasattr(tk, '_default_root'):
        root = tk.Tk()  # 使用ttkbootstrap创建的主窗口
    else:
        root = style.master
    app = LoginApp(root)
    root.mainloop()