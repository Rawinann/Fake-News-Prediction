# Fake-News-Prediction
1: Fake news
0: Real news

# การทำนายข่าวปลอม (Fake News Prediction)

โปรเจกต์นี้มีวัตถุประสงค์เพื่อพัฒนาโมเดลการเรียนรู้ของเครื่อง (Machine Learning) สำหรับการตรวจจับข่าวปลอม โดยใช้ข้อมูลที่มีการระบุว่าเป็น **ข่าวปลอม (Fake)** หรือ **ข่าวจริง (True)** จากนั้นจะทำการประมวลผลข้อมูลและฝึกโมเดลหลายตัวเพื่อหาโมเดลที่เหมาะสมที่สุด

---

## คุณสมบัติของโปรเจกต์

- **การประมวลผลข้อความ**: ทำความสะอาดข้อมูลข่าว เช่น การลบตัวอักษรพิเศษ ลิงก์ URL การลบคำที่ไม่สำคัญ (Stop Words) และการรวมข้อความระหว่าง `title` และ `text` 
- **การแปลงข้อความเป็นตัวเลข**: ใช้เทคนิค TF-IDF Vectorization เพื่อแปลงข้อความให้เป็นค่าตัวเลขที่สามารถป้อนเข้าสู่โมเดล Machine Learning ได้
- **การฝึกโมเดล**: ทดสอบและฝึกโมเดลหลายประเภท เช่น:
  - Logistic Regression
  - Random Forest Classifier
  - Gradient Boosting Classifier
  - XGBoost Classifier
- **การประเมินผลลัพธ์**: ใช้ค่าความแม่นยำ (Accuracy) และคะแนน Cross-Validation เพื่อประเมินประสิทธิภาพของโมเดล
- **การบันทึกไฟล์**: บันทึกโมเดลที่ฝึกสำเร็จและตัวแปลง TF-IDF Vectorizer ในรูปแบบ `.pkl` เพื่อใช้งานในอนาคต

---

## ข้อมูลที่ใช้ (Dataset)

1. **Fake.csv**: ไฟล์ที่มีข้อมูลข่าวปลอมพร้อมป้ายกำกับ
2. **True.csv**: ไฟล์ที่มีข้อมูลข่าวจริงพร้อมป้ายกำกับ

ข้อมูลทั้งสองจะถูกนำมารวมกัน สุ่มลำดับ และเตรียมพร้อมสำหรับการประมวลผลและการฝึกโมเดล

---

## โครงสร้างโปรเจกต์

```bash
fake-news-prediction/
├── app.py                     # สคริปต์สำหรับรันเว็บแอปพลิเคชัน
├── main.py                    # สคริปต์สำหรับการเทรนโมเดลและบันทึกโมเดล
├── True.csv                   # ข้อมูลข่าวจริง
├── Fake.csv                   # ข้อมูลข่าวปลอม
├── model_logistic_regression.pkl  # โมเดล Logistic Regression ที่บันทึกไว้
├── model_random_forest.pkl        # โมเดล Random Forest ที่บันทึกไว้
├── model_gradient_boosting.pkl    # โมเดล Gradient Boosting ที่บันทึกไว้
├── model_xgboost.pkl              # โมเดล XGBoost ที่บันทึกไว้
├── vectorizer.pkl                 # TF-IDF Vectorizer ที่บันทึกไว้
├── requirements.txt           # ไฟล์แสดงรายการไลบรารีที่ต้องติดตั้ง
├── README.md                  # คำอธิบายโปรเจกต์
```

---


## วิธีติดตั้ง

โปรแกรมนี้ต้องการ Python และไลบรารีที่เกี่ยวข้อง สามารถติดตั้งไลบรารีที่จำเป็นได้ด้วยคำสั่ง:

```bash
pip install -r requirements.txt
```

ไลบรารีที่จำเป็น:
pandas
numpy
nltk
scikit-learn
xgboost
joblib

กระบวนการทำงานของโปรเจกต์

**1. การประมวลผลข้อมูล**
รวมคอลัมน์ title และ text เป็นคอลัมน์ใหม่ชื่อ content
ทำความสะอาดข้อมูลโดย:
แปลงข้อความเป็นตัวพิมพ์เล็ก (Lowercase)
ลบตัวอักษรพิเศษ ลิงก์ URL และเครื่องหมายวรรคตอน
ลบคำที่ไม่สำคัญ (Stop Words) ด้วย NLTK

**2. การแปลงข้อความเป็นตัวเลข**
ใช้ TF-IDF Vectorization เพื่อแปลงข้อความเป็นค่าตัวเลข
จำกัดจำนวนคำสูงสุด 5,000 คำ
ใช้ n-grams ที่มีขนาด 1 ถึง 2 (Bi-grams)

**3. การฝึกและประเมินโมเดล**
ฝึกโมเดล Machine Learning ดังนี้:
Logistic Regression
Random Forest Classifier
Gradient Boosting Classifier
XGBoost Classifier
ประเมินโมเดลแต่ละตัวด้วย:
ค่าความแม่นยำ (Accuracy) บนชุดทดสอบ
คะแนน Cross-Validation บนชุดฝึก
บันทึกโมเดลทุกตัวและ TF-IDF Vectorizer เป็นไฟล์ .pkl
ผลลัพธ์ที่ได้

**ไฟล์โมเดลที่บันทึกไว้:** โมเดลที่ฝึกสำเร็จจะถูกบันทึกในรูปแบบ .pkl เช่น:
model_logistic_regression.pkl
model_random_forest.pkl
model_gradient_boosting.pkl
model_xgboost.pkl
ตัวแปลง TF-IDF: บันทึกเป็นไฟล์ vectorizer.pkl
โมเดลที่ดีที่สุด: ค้นหาโมเดลที่ดีที่สุดจากคะแนน Cross-Validation และแสดงผลใน Console
