import os
import re
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization

# --- AYARLAR ---
DATA_DIR = "./raw_data"
BATCH_SIZE = 64
VOCAB_SIZE = 15000
SEQUENCE_LENGTH = 20

# --- 1. VERİ YÜKLEME ---
def load_conversations(data_path):
    inputs, outputs = [], []
    
    # Klasör yoksa oluştur
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        print(f"UYARI: '{data_path}' klasörü yoktu, oluşturuldu. Lütfen içine .txt dosyaları at!")
        return [], []

    print(f">> '{data_path}' taranıyor...")
    
    # Klasördeki tüm .txt dosyalarını bul
    files = [f for f in os.listdir(data_path) if f.endswith('.txt')]
    
    if not files:
        print("HATA: Hiç dosya yok! Lütfen 'raw_data' klasörüne metin dosyaları koy.")
        return [], []

    total_lines = 0
    
    for filename in files:
        file_path = os.path.join(data_path, filename)
        print(f"   - Okunuyor: {filename}")
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                
            # Mantık: Satır 1=Soru, Satır 2=Cevap
            for i in range(0, len(lines) - 1, 2):
                q = lines[i].strip()
                a = lines[i+1].strip()
                
                if q and a:
                    inputs.append(q)
                    outputs.append(a)
                    total_lines += 1
        except Exception as e:
            print(f"   Hata: {filename} okunamadı. ({e})")

    print(f">> Toplam {total_lines} konuşma çifti yüklendi.")
    return inputs, outputs

# --- 2. TEMİZLİKÇİ ---
def custom_standardization(input_string):
    lowercased = tf.strings.lower(input_string)
    stripped = tf.strings.regex_replace(lowercased, "[^a-z0-9?! ]", "")
    return stripped

# --- 3. HAZIRLAYICI ---
def prepare_dataset(data_path=DATA_DIR):
    inputs, outputs = load_conversations(data_path)
    
    if not inputs:
        return None, None

    # Cevaplara start/end ekle
    outputs = ["[start] " + text + " [end]" for text in outputs]

    vectorizer = TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_mode="int",
        output_sequence_length=SEQUENCE_LENGTH,
        standardize=custom_standardization,
    )

    print(">> Kelimeler öğreniliyor...")
    vectorizer.adapt(inputs + outputs)
    
    text_ds = tf.data.Dataset.from_tensor_slices((inputs, outputs))
    text_ds = text_ds.shuffle(BATCH_SIZE * 10).batch(BATCH_SIZE)

    def vectorize_text(inputs, outputs):
        inputs = vectorizer(inputs)
        outputs = vectorizer(outputs)
        return ({"encoder_inputs": inputs, "decoder_inputs": outputs[:, :-1]}, outputs[:, 1:])

    dataset = text_ds.map(vectorize_text, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset, vectorizer

# --- TEST ---
if __name__ == "__main__":
    ds, vec = prepare_dataset()
    if ds:
        print("✅ data_loader.py başarıyla çalıştı!")
