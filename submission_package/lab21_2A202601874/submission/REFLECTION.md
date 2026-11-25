# Reflection — Lab 21

**Họ tên**: Nguyễn Xuân Phượng  
**Mã học viên**: 2A202601874

## 1. Điều gì làm tôi ngạc nhiên nhất?

Điều làm tôi ngạc nhiên nhất là baseline với optimized prompt đã đạt target 0.6875,
trong khi baseline với naive prompt chỉ đạt 0.0000. Điều đó cho thấy chất lượng prompt
đã tạo ra một mức chuẩn rất cao trước cả khi fine-tune. Fine-tune vẫn tăng target lên
0.9375, nhưng lại làm regression giảm, nên điểm target cao hơn chưa đủ để triển khai.

## 2. Tôi mất nhiều thời gian nhất ở đâu?

Tôi mất nhiều thời gian nhất ở bước huấn luyện và đánh giá trên GPU T4. T4 chỉ có 14.6
GB VRAM khả dụng và không hỗ trợ bfloat16, nên pipeline phải dùng fp16. Việc tải model
4B và chạy nhiều adapter cũng làm thời gian thực nghiệm dài hơn dự kiến.

## 3. Trước lab này tôi tin điều gì mà giờ không còn tin?

Trước lab, tôi nghĩ train loss thấp hơn gần như chắc chắn có nghĩa là model tốt hơn.
Sau thí nghiệm, tôi thấy `attn_only` có train loss thấp hơn `correct` nhưng target chỉ
hoà, còn regression mới là lý do khiến fine-tune bị FAILED. Vì vậy phải đánh giá trên
đúng task và kiểm tra cả các năng lực không nằm trong dataset fine-tune.

## 4. Tôi dùng AI assistant vào việc gì? Chỗ nào nó sai?

Tôi dùng AI assistant để đọc rubric, kiểm tra artefact, diễn giải kết quả và soạn report.
AI assistant hữu ích trong việc phát hiện rằng chạy đủ NB1–NB5 không đồng nghĩa với đánh
giá đủ dữ liệu, vì `EVAL_LIMIT=8` vẫn giới hạn NB2 và NB5. Phần cần kiểm tra thủ công là
trạng thái runtime Colab và số mẫu thật sự được ghi trong JSON.

## 5. Nếu ngày mai fine-tune cho khách hàng thật, bước đầu tiên là gì?

Bước đầu tiên là xác định task metric, tập regression và baseline prompt trước khi train.
Sau đó tôi sẽ kiểm tra loss mask trên một vài mẫu bằng token thực tế, đóng băng eval set,
đo baseline rồi mới huấn luyện. Tôi cũng sẽ thêm dữ liệu replay cho năng lực chung để
tránh model chuyên môn hóa quá mức và làm suy giảm các khả năng sẵn có.
