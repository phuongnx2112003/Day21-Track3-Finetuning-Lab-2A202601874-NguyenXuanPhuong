# Lab 21 — Evaluation Report

**Họ tên**: Nguyễn Xuân Phượng  **MSSV**: 2A202601874  **Ngày**: 2026-08-21  
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  
**GPU thực tế**: Tesla T4, CUDA `sm_75`, 14.6 GB, precision `fp16`

> Lưu ý về tính hợp lệ: các artefact trong ZIP được sinh với `EVAL_LIMIT=8`, vì vậy
> kết quả dưới đây là smoke evaluation trên 8 mẫu target và 8 mẫu regression, chưa phải
> kết quả full 50/15 để nộp chính thức. Kết luận được giữ nguyên trung thực theo artefact.

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường |
| Train / val | 225 / 25, seed 42 |
| `max_length` | 1024 — p95 đo được là 98 token; gợi ý theo p95 là 256 |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epochs / 30 optimizer steps |

T4 không hỗ trợ bfloat16 nên pipeline tự chọn fp16. `max_length=1024` là cấu hình của
tier T4, nhưng số đo dữ liệu cho thấy p95 chỉ là 98 và giá trị gợi ý là 256. Vì vậy,
1024 an toàn về mặt cắt chuỗi nhưng dư thừa về padding và thời gian; một lần chạy tối ưu
hơn nên dùng 256, sau khi kiểm tra lại toàn bộ pipeline.

Chat template có giữ khối `<think>`: **có**. `template_check.json` cho thấy reasoning
được giữ lại trong chuỗi render, vì vậy template không âm thầm xoá trace. Tuy nhiên
`valid_trace_rate=0.0` trong eval; corpus ticket hiện tại chủ yếu có câu trả lời JSON
không chứa reasoning trace thực tế.

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | `0.4149` (39/94 token) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Đoạn đầu được tính loss là:

```text
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Đối chứng `everything` cho 94/94 token được supervised, bao gồm cả system prompt và
câu hỏi. Điều này xác nhận vì sao `assistant-only` là lựa chọn đúng: model học câu trả
lời JSON thay vì học cách lặp lại prompt.

## 3. Ba baseline (NB2 — đo trước khi train)

| Run | target | regression | format | latency (ms) |
|---|---:|---:|---:|---:|
| (a) base + naive prompt | 0.0000 | 0.7500 | 0.0000 | 3283.3 |
| (b) base + optimized prompt | 0.6875 | 0.7500 | 1.0000 | 1059.8 |
| (c) LoRA fine-tune | 0.9375 | 0.6250 | 1.0000 | 1506.0 |

Baseline (b) mạnh hơn rõ rệt baseline (a): target tăng từ 0.0000 lên 0.6875, format
tăng từ 0 lên 1.0 và latency giảm từ 3283.3 ms xuống 1059.8 ms. Prompt optimized
được giữ nguyên sau khi đóng băng; SHA được ghi nhận là `719e74d3b6232053`.

Fine-tune tiếp tục tăng target lên 0.9375, nhưng chậm hơn baseline (b) khoảng 42.1%
và làm regression giảm từ 0.7500 xuống 0.6250. Vì vậy, target accuracy cao hơn chưa
đủ để kết luận bản fine-tune tốt hơn cho deployment.

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss | target (NB5) | train s | VRAM GB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.6265 | 0.9375 | 969.5 | 12.01 |
| `attn_only` | q,v | 283 | 32,456,704 | 1e-4 | 0.5371 | 0.9375 | 781.9 | 12.02 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 1.5704 | 0.0000 | 926.7 | 12.01 |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 | 0.7058 | 0.8438 | 983.5 | 7.09 |

### 4.1 — Vị trí adapter và rank

`attn_only` dùng rank 283 để có ngân sách gần như bằng `correct`: 32,456,704 so với
32,464,896 tham số, chênh lệch chỉ khoảng 0.025%. Trên tập target smoke, hai run hoà
nhau ở 0.9375. Tuy nhiên train loss của `attn_only` thấp hơn `correct` (0.5371 so với
0.6265). Như vậy thứ tự theo train loss không cung cấp bằng chứng rằng attention-only
tốt hơn trên tác vụ; rank cao hơn có thể làm loss thấp hơn, còn vị trí adapter chưa tạo
ra khác biệt target ở mẫu thử này. Đây là lý do phải match ngân sách và xếp hạng bằng
task metric thay vì chỉ nhìn train loss.

### 4.2 — Learning rate

`wrong_lr` chỉ thay learning rate từ `1e-4` xuống `1e-5`. Kết quả train loss tăng lên
1.5704, cao hơn nhiều so với `correct` 0.6265, và target/format đều giảm về 0.0 trên
8 mẫu. Learning rate quá thấp khiến 30 step chưa đủ để adapter học quy tắc đầu ra.
Nếu chỉ nhìn một vài log đầu hoặc chỉ nhìn thời gian chạy, có thể tưởng rằng run đã
hoàn thành bình thường; nhưng task score cho thấy nó không học được nhiệm vụ. Ngược
lại, train loss thấp cũng không tự động chứng minh khả năng tổng quát tốt.

### 4.3 — QLoRA

QLoRA giảm VRAM từ 12.01 GB xuống 7.09 GB, tiết kiệm khoảng 4.92 GB, tương đương
40.97%. Đổi lại, train loss cao hơn `correct` (0.7058 so với 0.6265), target thấp hơn
(0.8438 so với 0.9375) và latency cao hơn (1780.2 ms so với 1506.0 ms). Trên model
Qwen3.5-4B này, kết quả smoke ủng hộ khuyến nghị ưu tiên LoRA 16-bit khi T4 còn đủ
VRAM; QLoRA là phương án tiết kiệm bộ nhớ, nhưng phải chấp nhận chất lượng và tốc độ
kém hơn trong phép đo này.

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy: `FAILED`**  
`target Δ = +0.250` · `regression Δ = -0.125` · `valid_trace_rate = 0.000`

Fine-tune thắng baseline prompt optimized trên target với mức tăng 0.250, từ 0.6875
lên 0.9375. Format vẫn đạt 1.0, nên model đã học đúng hình thức JSON và cải thiện đáng
kể khả năng triage ticket. Tuy nhiên regression giảm 0.125, từ 0.7500 xuống 0.6250,
vượt xa ngưỡng cho phép 0.020. Điều này cho thấy adapter đã chuyên môn hóa theo ticket
CSKH nhưng làm suy giảm khả năng trả lời kiến thức phổ thông. Vì vậy cổng hồi quy phải
đánh dấu FAILED dù target tăng. Kết quả này không có nghĩa fine-tune vô dụng; nó cho
thấy cần thêm replay data phổ thông khoảng 1–5% hoặc một chiến lược regularization để
giữ năng lực nền trước khi triển khai.

## 6. Định tính

Artefact `qualitative.json` chỉ lưu ticket rút gọn, điểm field-level và phần đầu output
fine-tune; nó không lưu đầy đủ nhãn vàng hoặc output baseline (b). Vì vậy bảng dưới đây
không bịa nhãn/baseline còn thiếu, mà dùng đúng bằng chứng đã được lưu.

| # | Ticket rút gọn | FT field score | FT output quan sát được | Nhận xét |
|---:|---|---:|---|---|
| 1 | Bình giữ nhiệt — chưa thấy tiền | 0.75 | `intent=hoan_tien`, `product=bình giữ nhiệt` | FT sai ít nhất một trường |
| 2 | Nồi chiên không dầu — thiếu phụ kiện | 0.75 | `intent=san_pham_loi`, `product=nồi chiên không dầu` | FT sai ít nhất một trường |
| 3 | Chuột không dây — muốn trả lại, gấp | 1.00 | JSON triage đúng các trường được chấm | FT đạt điểm tuyệt đối |
| 4 | Ốp lưng — hoàn tiền, sớm, bực mình | 1.00 | JSON triage đúng các trường được chấm | FT đạt điểm tuyệt đối |
| 5 | Đèn bàn LED — vỡ khi nhận, gấp | 1.00 | JSON triage đúng các trường được chấm | FT đạt điểm tuyệt đối |

Hai ca điểm 0.75 cho thấy lỗi còn lại nằm ở việc phân biệt các thuộc tính phụ như
urgency hoặc sentiment, không phải việc trích xuất product. Do file hiện tại không lưu
gold label và prediction của baseline (b), cần chạy một phiên bản qualitative bổ sung
nếu muốn báo cáo trực tiếp FT thắng/thua so với baseline trên từng ticket.

## 7. Kết luận & điều tôi học được

Với kết quả smoke hiện tại, tôi chưa nên deploy bản fine-tune dù target score đã tăng
từ 0.6875 lên 0.9375 so với baseline prompt optimized. Lý do quyết định là regression
giảm từ 0.7500 xuống 0.6250, cho thấy model có nguy cơ đánh đổi năng lực chung để học
quy tắc triage. Ngoài ra, fine-tune có latency 1506.0 ms/mẫu, cao hơn baseline prompt
optimized 1059.8 ms/mẫu, nên lợi ích target chưa hoàn toàn miễn phí. Pipeline đúng ở
phần mask: chỉ 41.49% token được supervised, câu trả lời nằm trong loss và câu hỏi bị
mask. Thí nghiệm cũng cho thấy learning rate là đòn bẩy mạnh: giảm 10 lần làm target
và format về 0. Vị trí adapter và rank cần được đánh giá bằng task metric; train loss
thấp hơn của `attn_only` không giúp nó vượt `correct` trên smoke target. Nếu có thêm
thời gian, hướng hợp lý nhất là thêm 1–5% replay data regression, chạy lại full eval
50/15, sau đó so sánh lại target, regression, format và latency trước khi quyết định
deploy.

Ba điều tôi học được:

1. Mask phải được kiểm chứng bằng token thực tế; `assistant-only` cho 39/94 token được
   supervised, còn `everything` sẽ tính loss cả prompt.
2. Train loss không phải task metric: `attn_only` có loss thấp hơn nhưng target chỉ hoà
   `correct`, còn `wrong_lr` cho thấy learning rate sai có thể phá hỏng toàn bộ run.
3. Một fine-tune tăng target vẫn có thể không deploy được nếu làm regression giảm; cần
   đánh giá cả năng lực chuyên môn, năng lực chung, format và latency.

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng
- [ ] B3 reasoning-trace collapse
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub

## Ghi chú trước khi nộp

Kết quả hiện tại là smoke run (`n=8`). Cần unset `EVAL_LIMIT` và chạy lại NB2 + NB5 để
report chính thức dùng đủ 50 target và 15 regression mẫu. Sau đó thay các số liệu trong
report bằng artefact full mới và điền họ tên/MSSV.
