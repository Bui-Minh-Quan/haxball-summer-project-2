# Haxball 2D AI — Autonomous RL Soccer Game
![alt text](assets/images/for_readme/in_game.png)

## 1. Giới thiệu

* **Haxball là gì?** Tựa game thể thao 2D góc nhìn trên xuống kết hợp giữa bóng đá và air-hockey. Hai đội (Đỏ và Xanh) điều khiển các đĩa tròn di chuyển theo 8 hướng và sút bóng vào khung thành đối phương trong phạm vi thời gian hoặc điểm số quy định.
* **Dự án thực hiện những gì?**
  * Tự xây dựng engine vật lý 2D (sub-stepped circle collisions, phản xạ đàn hồi, co giãn tỷ lệ sân động).
  * Phát triển và thử nghiệm hai kiến trúc mạng nơ-ron: MLP và Entity-Transformer.
  * Huấn luyện các agent bóng đá tự hành bằng thuật toán Multi-Agent PPO (MAPPO) kết hợp Curriculum Learning và Self-Play qua hơn 300 triệu bước môi trường.
  * Tích hợp trực tiếp các mô hình đã huấn luyện vào game thông qua ONNX Runtime để xử lý suy luận mượt mà theo thời gian thực.

---

## 2. Cài đặt và Khởi chạy

Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

Khởi chạy trò chơi:
```bash
python main.py
# (Trên Linux: python3 main.py)
```

---

## 3. Demo & Tính năng trò chơi

* **Link video demo:** [Xem demo trên Google Drive](https://drive.google.com/file/d/1bjRnX3Dg2ATI4LlLKR-awUe0IrwyxiJG/view?usp=sharing)

Chế độ **Chơi nhanh (Quick Play)** cho phép tùy chỉnh linh hoạt:

* **Thể thức thi đấu**: 1v1 (Solo), 2v2 (Duo), 3v3 (Squad).
* **Phe thi đấu**: Đội Đỏ (Red) hoặc Đội Xanh (Blue).
* **Thời gian trận đấu**: Vô hạn hoặc từ 1 đến 15 phút.
* **Giới hạn điểm**: Vô hạn hoặc từ 1 đến 15 bàn thắng.
* **Kích thước sân**: Sân nhỏ (`Small`), Truyền thống (`Classic`), Sân lớn (`Big`), Khổng lồ (`Huge`).

### Hệ thống đối thủ

* **Heuristic Bot**: Đối thủ điều khiển theo luật hình học giải tích (rule-based) với khả năng chuyển đổi linh hoạt giữa các trạng thái chiến thuật (tấn công, phòng thủ, cân bằng) và kiểm soát quán tính khung thành.
* **MLP Agent**: Agent học tăng cường sử dụng mạng nơ-ron Perceptron đa tầng (MLP), đưa ra quyết định dựa trên vector quan sát phẳng được chuẩn hóa.
* **Transformer Agent**: Agent học tăng cường đa tác nhân xây dựng trên kiến trúc Entity-Transformer kết hợp frame-stacking, có khả năng xử lý tương tác động giữa các thực thể và phối hợp đồng đội ở cả 3 thể thức 1v1, 2v2 và 3v3.

![alt text](assets/images/for_readme/config.png)

---

## 4. Cấu trúc dự án

```text
.
├── assets/                  # Sprites bóng, sân cỏ, font retro, audio & mô hình ONNX (.onnx)
├── config/                  # Cấu hình game, thông số vật lý và thiết lập trận đấu
├── Notebooks/
│   ├── training_2/          # Quá trình huấn luyện & checkpoints cho kiến trúc MLP
│   └── training_transformer/# Quá trình huấn luyện & checkpoints cho kiến trúc Entity-Transformer
├── src/
│   ├── bots/                # Heuristic AI điều khiển theo luật hình học
│   ├── engine/              # Physics 2D, mô phỏng thực thể, sân bãi & controller cơ bản
│   ├── game/                # Quản lý đồ họa Pygame, camera tracking, FSM & UI
│   └── rl/                  # Pipeline huấn luyện và môi trường cho mô hình MLP
│   └── rl_transformer/      # Pipeline huấn luyện MAPPO và môi trường cho Entity-Transformer
├── tools/                   # Script xuất và kiểm thử sai số mô hình (PyTorch -> ONNX)
├── main.py                  # Điểm khởi chạy trò chơi
└── requirements.txt         # Danh sách thư viện phụ thuộc
```