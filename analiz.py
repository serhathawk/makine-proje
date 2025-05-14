import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# NLTK kaynaklarını indirmek için (ilk çalıştırmada gerekebilir)
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4') # WordNet için ek kaynak

# 1. Veri Yükleme
try:
    df = pd.read_csv('chatbot_intent_classification.csv')
except FileNotFoundError:
    print("HATA: 'chatbot_intent_classification.csv' dosyası bulunamadı.")
    print("Lütfen dosyanın kodla aynı dizinde olduğundan emin olun veya tam dosya yolunu belirtin.")
    exit()

print("Veri Seti İlk 5 Satır:")
print(df.head())
print("\nVeri Seti Bilgisi:")
df.info()
print("\nNiyet Sınıflarının Dağılımı:")
print(df['intent'].value_counts())

# 2. Veri Ön İşleme Fonksiyonları
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Küçük harfe dönüştürme
    text = text.lower()
    # Noktalama işaretlerini ve özel karakterleri kaldırma (sayıları da kaldırır)
    text = re.sub(r'[^a-z\s]', '', text) # Sadece harfleri ve boşlukları tutar
    # Tokenizasyon
    tokens = word_tokenize(text)
    # Stopword temizliği ve Lemmatization
    processed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 1]
    return " ".join(processed_tokens)

print("\nVeri Ön İşleme Başlatılıyor...")
df['processed_input'] = df['user_input'].apply(preprocess_text)
print("Veri Ön İşleme Tamamlandı.")
print("\nİşlenmiş Veri Seti İlk 5 Satır:")
print(df[['user_input', 'processed_input', 'intent']].head())

# 3. Özellik Çıkarımı (TF-IDF)
print("\nTF-IDF Vektörleştirme Başlatılıyor...")
tfidf_vectorizer = TfidfVectorizer(max_features=1000) # En sık geçen 1000 kelimeyi kullan
X = tfidf_vectorizer.fit_transform(df['processed_input'])
y = df['intent']
print("TF-IDF Vektörleştirme Tamamlandı.")
print("Öznitelik Matrisi Boyutu:", X.shape)

# 4. Veri Setini Eğitim ve Test Olarak Ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nEğitim Seti Boyutu: {X_train.shape[0]} örnek")
print(f"Test Seti Boyutu: {X_test.shape[0]} örnek")

# 5. Modelleri Tanımlama ve Eğitme
models = {
    "Naive Bayes": MultinomialNB(),
    "Lojistik Regresyon": LogisticRegression(max_iter=1000, random_state=42), # max_iter artırıldı
    "SVM (Linear Kernel)": SVC(kernel='linear', random_state=42)
}

results = {}

print("\nModel Eğitimi ve Değerlendirme Başlatılıyor...")
for model_name, model in models.items():
    print(f"\n--- {model_name} ---")
    # Model Eğitimi
    model.fit(X_train, y_train)
    # Tahminler
    y_pred = model.predict(X_test)
    # Değerlendirme
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    results[model_name] = {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": cm,
        "predictions": y_pred 
    }
    print(f"Doğruluk (Accuracy): {accuracy:.4f}")
    print("Sınıflandırma Raporu:")
    print(report)
    print("Karmaşıklık Matrisi:")
    print(cm)


print("\nTüm Modellerin Sonuçları results değişkeninde saklandı.")
