# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Khi temperature tăng từ 0.0 lên 1.5, câu trả lời chuyển từ rập khuôn, trực diện sang bay bổng và sáng tạo hơn. Tuy nhiên, ở mức quá cao như 1.5, văn bản bắt đầu trở nên lủng củng, dùng từ ngữ kỳ lạ, mất đi sự mạch lạc hoặc sinh ra ảo giác

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Mình sẽ đặt temperature ở mức thấp (khoảng 0.0 - 0.2). Lý do là vì chatbot hỗ trợ khách hàng cần trả lời chính xác, nhất quán theo tài liệu của công ty, không được "sáng tạo" hay bịa ra thông tin sai lệch để tránh gây hiểu nhầm cho khách.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> GPT-4o đắt hơn GPT-4o-mini khoảng 16.7 lần (0.010 / 0.0006 = 16.67). 
> Trường hợp GPT-4o xứng đáng: Các tác vụ cần tư duy logic phức tạp, viết code khó, hoặc phân tích số liệu chuyên sâu.
> Trường hợp nên dùng mini: Các tác vụ đơn giản như tóm tắt văn bản, trích xuất dữ liệu chuẩn form, dịch thuật cơ bản hoặc chatbot tán gẫu thông thường để tối ưu chi phí và tốc độ

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> - Ở vai giáo viên tiểu học, phản hồi ngắn gọn, dùng từ ngữ đơn giản, gần gũi, nhiều ví dụ liên tưởng (ví dụ: ví blockchain như quyển sổ tay chung của lớp không ai xóa được).
> - Ở vai chuyên gia tài chính, model trả lời dài hơn, sử dụng thuật ngữ chuyên ngành (sổ cái phân tán, mã hóa, node mạng).
> System prompt có sức mạnh định hình rõ ràng phong cách, độ phức tạp và hướng tiếp cận của câu trả lời mà không cần thay đổi user prompt.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Khi lấy một đoạn văn tiếng Việt có 111 từ: 
> - Ước lượng thô (từ / 0.75): khoảng 148 token.
> - Đếm thực tế bằng tiktoken: 131 token.
> Hai con số chênh nhau khoảng 11-15% tùy vào nội dung đoạn văn. Tiếng Việt thường tốn nhiều token hơn tiếng Anh (có thể gấp đôi) vì bộ tokenizer của OpenAI được train chủ yếu trên dữ liệu tiếng Anh. Một từ tiếng Anh thường gom thành 1 token, nhưng một từ tiếng Việt do có dấu và dấu cách xen kẽ (từ ghép) nên thường bị cắt vỡ ra thành 2-3 subword (token) khác nhau.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất trong các ứng dụng tương tác trực tiếp với người dùng (như ChatGPT), giúp giảm cảm giác chờ đợi vì người dùng có thể đọc ngay những chữ đầu tiên thay vì nhìn màn hình xoay vòng vòng mấy chục giây. Ngược lại, non-streaming phù hợp hơn với các tác vụ xử lý ngầm ở hệ thống hậu cảnh (chẳng hạn như bóc tách dữ liệu ra JSON, tóm tắt tự động, dịch thuật văn bản ngầm) vì khi đó ta chỉ quan tâm kết quả cuối cùng một cách toàn vẹn để chuyển sang bước xử lý tiếp theo chứ không cần xem quá trình tạo ra chữ.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff giúp giảm tải dần dần cho máy chủ khi nó đang gặp sự cố hoặc quá tải. Nếu hàng nghìn client cùng dùng delay cố định (chờ 1 giây), chúng sẽ cùng đồng loạt tấn công lại máy chủ vào đúng 1 giây sau đó, tạo thành hiệu ứng "đàn trâu giẫm đạp" (Thundering Herd) khiến máy chủ chưa kịp phục hồi lại tiếp tục bị sập. Việc thời gian chờ tăng gấp đôi qua mỗi lần (0.1, 0.2, 0.4, 0.8...) sẽ phân tán các lượng truy cập này ra, giúp server "có không gian để thở" và dần dần xử lý được request.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Persona: "Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn bằng tiếng Việt."
> Giải thích: 
> 1. "ngắn gọn": Do chatbot chạy trên giao diện dòng lệnh (Terminal) nhỏ hẹp, nếu AI trả lời quá dài sẽ làm trôi chữ, khó đọc và gây chậm.
> 2. "tiếng Việt": Cố định ngôn ngữ để mô hình không bị "loạn" (tự động chuyển sang nói tiếng Anh) khi người dùng hỏi các câu có chứa nhiều thuật ngữ tiếng Anh.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất: Trợ lý chỉ lưu được 3 lượt chat gần nhất (6 tin nhắn), hoàn toàn không có bộ nhớ dài hạn. Nếu người dùng hỏi lại một vấn đề từ đầu buổi chat, model sẽ không nhớ gì cả.
> Đề xuất cải thiện: Thêm cơ chế "Tóm tắt định kỳ" (Summarization). Cụ thể, khi lịch sử đạt mốc 6 tin nhắn, thay vì xóa bỏ vĩnh viễn 2 tin nhắn cũ, ta gọi API một lần nữa yêu cầu model "Tóm tắt ngắn gọn các ý chính của đoạn chat này". Tóm tắt đó sẽ được đưa lên đầu làm ngữ cảnh nền thay thế cho tin nhắn cũ, giúp AI nhớ được quá khứ dài hạn mà vẫn tiết kiệm Token.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
