---
title: Transformer Networks in NLP and Vision
markmap:
  maxWidth: 300
  initialExpandLevel: 2
---

# Transformer Networks

## Introduction

- **Transformer Neural Network**, hay ngắn gọn là **Transformer**, là một kiến trúc mạng nơ-ron được giới thiệu năm 2017 trong bài báo nổi tiếng [“Attention is All You Need”](https://arxiv.org/abs/1706.03762). Tiêu đề này nói đến **cơ chế attention**, nền tảng cho việc xử lý dữ liệu bằng Transformer.

- Các Transformer Network hiện là loại mô hình Deep Learning chiếm ưu thế trong lĩnh vực NLP. Chúng đã thay thế hoàn toàn Recurrent Neural Networks trong mọi tác vụ NLP, và tất cả **Large Language Models (LLMs)** hiện nay đều dựa trên kiến trúc Transformer. Gần đây, Transformer cũng được mở rộng sang các lĩnh vực khác như xử lý ảnh, video, dự đoán chuỗi protein và DNA, xử lý chuỗi thời gian, và cả học tăng cường (reinforcement learning).
  → Transformer hiện là **kiến trúc mạng nơ-ron quan trọng nhất** trong AI.

---

## Historical Context

- Khái niệm tiên phong này không chỉ mang tính lý thuyết mà còn được ứng dụng thực tế, đặc biệt qua gói [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor) của TensorFlow.
  Nhóm Harvard NLP cũng đóng góp quan trọng khi cung cấp [hướng dẫn chú thích cho bài báo](https://nlp.seas.harvard.edu/2018/04/03/attention.html) cùng với bản cài đặt bằng PyTorch.
  Bạn có thể xem thêm hướng dẫn [xây dựng Transformer từ đầu bằng PyTorch](https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch).

- Sự ra đời của Transformer đã mở ra kỷ nguyên mới trong AI — thường được gọi là **Transformer AI** — nền tảng cho các đột phá tiếp theo như **BERT**.
  Đến năm 2018, nó đã được xem như một bước ngoặt của NLP.
  Năm 2020, OpenAI công bố [GPT-3](https://arxiv.org/abs/2005.14165), mô hình nhanh chóng gây tiếng vang khi có thể viết thơ, lập trình, sáng tác nhạc, tạo trang web, và hơn thế nữa.

- Năm 2021, các học giả Stanford gọi các mô hình này là [**foundation models**](https://www.datacamp.com/blog/what-are-foundation-models), nhấn mạnh vai trò nền tảng của Transformer trong việc định hình lại trí tuệ nhân tạo.

- > “Chúng ta đang sống trong thời kỳ mà những phương pháp đơn giản như mạng nơ-ron đang mở ra hàng loạt khả năng mới.”
- > — _Ashish Vaswani_, đồng tác giả bài báo gốc, cựu nhà nghiên cứu cao cấp tại Google.

---

## Self-Attention Mechanism

- **Self-attention** là cơ chế giúp mô hình tập trung vào những phần quan trọng của dữ liệu khi dự đoán.
  Trong NLP, self-attention giúp xác định những từ có liên quan đến từ đang xét trong một câu.

- Ví dụ: trong hai câu dưới đây, từ _“it”_ có thể chỉ “street” trong câu thứ nhất, nhưng chỉ “animal” trong câu thứ hai.
- Transformer mô hình hóa mối quan hệ giữa mọi từ trong câu và gán **trọng số (attention score)** cho từng cặp từ.

- ![Self Attention Example 1](https://lms.fit.hanu.vn/pluginfile.php/1/core_h5p/content/3071/images/image-JvuSAo7A.png)

- Cơ chế này tính **độ tương quan (attention score)** giữa mỗi cặp từ bằng **tích vô hướng (dot product)** giữa các vector biểu diễn từ đó:
  với mỗi **Query (Q)** và **Key (K)**, attention score = **Q·K**.

-![Self Attention Example 2](https://lms.fit.hanu.vn/pluginfile.php/1/core_h5p/content/3071/images/image-y3cTNNie.png)

- Sau đó, các giá trị này được chia cho √d (với _d_ là kích thước embedding) rồi chuẩn hóa bằng hàm **softmax** để đưa về khoảng [0, 1]:

- ![Attention Formula](https://lms.fit.hanu.vn/pluginfile.php/1/core_h5p/content/3071/images/image-hYSNlXoT.png)

- Quá trình nhân với **Value (V)** của các từ cho ra đầu ra của module self-attention.

---

- Self-attention trong Transformer hoạt động như một hệ thống tìm kiếm:
  mỗi từ đóng vai trò **query**, so sánh với các **key**, và nhận về **value** tương ứng — giống như việc truy vấn trong công cụ tìm kiếm.

- ![Self-Attention Architecture](https://lms.fit.hanu.vn/pluginfile.php/1/core_h5p/content/3071/images/image-AZYPfvVN.png)

- Kết quả: Transformer hiểu nghĩa của từ dựa trên **ngữ cảnh toàn câu**, cập nhật embedding của từ để phản ánh mối quan hệ đó.

---

## Multi-Head Attention

- Transformer bao gồm nhiều mô-đun self-attention song song, gọi là **attention heads**.
- Tổng hợp đầu ra của các attention heads tạo thành **multi-head attention**.

Ví dụ:

- Mô hình Transformer gốc có 8 heads.
- GPT-3 có 12 heads.

![Multi-Head Attention](https://lms.fit.hanu.vn/pluginfile.php/1/core_h5p/content/3071/images/image-c0CxlER5.png)

Lợi ích:

- Mỗi head học **một kiểu quan hệ ngữ nghĩa khác nhau** giữa các từ (ví dụ: danh từ – số lượng, tính từ – danh từ, v.v.).
- Các heads chạy **song song**, giúp **tăng tốc và mở rộng quy mô** huấn luyện.

---

## Encoder Block

- **Encoder Block** xử lý các embedding đầu vào và trích xuất biểu diễn đặc trưng cho các tác vụ NLP khác nhau.

Các thành phần chính:

- **Multi-head Attention layer** — nhiều self-attention heads.
- **Dropout layer** — giảm overfitting.
- **Residual connections** — kết nối tắt (skip connection), cải thiện ổn định gradient.
- **Layer Normalization** — chuẩn hóa đầu ra từng layer (0 mean, 1 std).
- **Feed Forward network** — gồm 2 lớp Dense, trích xuất đặc trưng phi tuyến.

- Mô hình Transformer gốc sử dụng **6 encoder blocks** nối tiếp.

![Encoder Block](https://lms.fit.hanu.vn/pluginfile.php/1/core_h5p/content/3071/images/image-crgvsBRJ.png)

### Ví dụ code Keras/TensorFlow:

```python
import tensorflow as tf
from tensorflow import keras
from keras.layers import MultiHeadAttention, LayerNormalization, Dropout, Dense, Embedding, Layer
from keras import Sequential

class TransformerEncoder(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.feed_forward_net = Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layer_normalization1 = LayerNormalization(epsilon=1e-6)
        self.layer_normalization2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        att_out = self.multi_head_attention(inputs, inputs)
        att_out = self.dropout1(att_out, training=training)
        add_norm = self.layer_normalization1(inputs + att_out)
        ffn_out = self.feed_forward_net(add_norm)
        ffn_out = self.dropout2(ffn_out, training=training)
        return self.layer_normalization2(add_norm + ffn_out)
```

---

## Positional Encoding

- Transformer sử dụng **word embeddings**, nhưng embedding bản thân **không chứa thông tin về thứ tự từ**.
- Do đó, cần thêm **positional encoding** để mô hình hiểu ngữ pháp theo vị trí.

![Positional Encoding](https://lms.fit.hanu.vn/pluginfile.php/1/core_h5p/content/3071/images/image-35lH03OD.png)

- Bản gốc dùng **sin/cos positional encoding** (vector tuần hoàn có giá trị từ -1 đến 1).
- Ở đây ta minh họa cách dùng **learned positional embedding** — mô hình học vị trí như học từ vựng.

```python
class TokenAndPositionEmbedding(Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_embeddings = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.positional_embeddings = Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, inputs):
        maxlen = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        pos_embeddings = self.positional_embeddings(positions)
        tok_embeddings = self.token_embeddings(inputs)
        return tok_embeddings + pos_embeddings
```

---

# Vision Transformers (ViT)

- Sau thành công của Transformer trong NLP, mô hình này đã được mở rộng sang **Computer Vision**.
- Mô hình đầu tiên (2021) là **Vision Transformer – ViT**.

![ViT Overview](https://lms.fit.hanu.vn/pluginfile.php/1/core_h5p/content/3071/images/image-gXKgAVT8.gif)

### Cách hoạt động:

- Ảnh được chia thành các **patch nhỏ (16×16 pixel)**, mỗi patch là một “token”.
- Mỗi patch được **làm phẳng (flatten)** rồi qua **dense layer** để tạo embedding.
- Thêm **positional embedding** và **class embedding**.
- Chuỗi patch embedding được đưa vào **encoder block** tiêu chuẩn của Transformer.

---

### Kích thước các phiên bản ViT

- **Base**: 12 encoder blocks, 768 chiều, 86M tham số
- **Large**: 24 blocks, 1024 chiều, 307M tham số
- **Huge**: 32 blocks, 1280 chiều, 632M tham số

---

### Các biến thể ViT khác

- Các phiên bản mở rộng sau này bao gồm:

- **MaxViT** (Multi-axis ViT)
- **Swin** (Shifted Window ViT)
- **DeiT** (Data-efficient Image Transformer)
- **T2T-ViT** (Token-to-token ViT)

- Các mô hình này đạt **độ chính xác cao hơn CNN truyền thống** (EffNet, ConvNeXt, NFNet) trên nhiều tác vụ thị giác.

![ViT Accuracy on ImageNet](https://lms.fit.hanu.vn/pluginfile.php/1/core_h5p/content/3071/images/image-1oa7fCEP.png)
