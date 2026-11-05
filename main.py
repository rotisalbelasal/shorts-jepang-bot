import os
import sys
import json
import time
import glob
import random
import shutil
import subprocess
import urllib.request
import urllib.parse
from datetime import datetime, timezone
from PIL import Image, ImageDraw, ImageFont

# =====================================================================
# KONFIGURASI
# =====================================================================
STATE_DIR = "state"
HISTORY_FILE = os.path.join(STATE_DIR, "history.json")
HEARTBEAT_FILE = os.path.join(STATE_DIR, "heartbeat.txt")
WORK_DIR = "build_temp"
OUTPUT_VIDEO = "final_short.mp4"

WIDTH = 1080
HEIGHT = 1920
BG_COLOR = (0, 0, 0)
TEXT_COLOR = (255, 255, 255)
WORDS_PER_VIDEO = 5
REPEATS_PER_WORD = 5
FPS = 24

os.makedirs(STATE_DIR, exist_ok=True)
if os.path.exists(WORK_DIR):
    shutil.rmtree(WORK_DIR)
os.makedirs(WORK_DIR, exist_ok=True)

# Pastikan file state selalu ada
if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)
if not os.path.exists(HEARTBEAT_FILE):
    with open(HEARTBEAT_FILE, "w", encoding="utf-8") as f:
        f.write("init\n")

# =====================================================================
# LAPIS 3: BANK KOSAKATA LOKAL (SUDAH DIBERSIHKAN)
# =====================================================================
OFFLINE_VOCAB_BANK = [
    {"jp": "食べる (Taberu)", "tts_jp": "たべる", "id": "Makan"},
    {"jp": "飲む (Nomu)", "tts_jp": "のむ", "id": "Minum"},
    {"jp": "行く (Iku)", "tts_jp": "いく", "id": "Pergi"},
    {"jp": "来る (Kuru)", "tts_jp": "くる", "id": "Datang"},
    {"jp": "見る (Miru)", "tts_jp": "みる", "id": "Melihat"},
    {"jp": "聞く (Kiku)", "tts_jp": "きく", "id": "Mendengar"},
    {"jp": "話す (Hanasu)", "tts_jp": "はなす", "id": "Berbicara"},
    {"jp": "読む (Yomu)", "tts_jp": "よむ", "id": "Membaca"},
    {"jp": "書く (Kaku)", "tts_jp": "かく", "id": "Menulis"},
    {"jp": "買う (Kau)", "tts_jp": "かう", "id": "Membeli"},
    {"jp": "働く (Hataraku)", "tts_jp": "はたらく", "id": "Bekerja"},
    {"jp": "休む (Yasumu)", "tts_jp": "やすむ", "id": "Istirahat"},
    {"jp": "寝る (Neru)", "tts_jp": "ねる", "id": "Tidur"},
    {"jp": "起きる (Okiru)", "tts_jp": "おきる", "id": "Bangun"},
    {"jp": "帰る (Kaeru)", "tts_jp": "かえる", "id": "Pulang"},
    {"jp": "待つ (Matsu)", "tts_jp": "まつ", "id": "Menunggu"},
    {"jp": "作る (Tsukuru)", "tts_jp": "つくる", "id": "Membuat"},
    {"jp": "使う (Tsukau)", "tts_jp": "つかう", "id": "Menggunakan"},
    {"jp": "知る (Shiru)", "tts_jp": "しる", "id": "Mengetahui"},
    {"jp": "考える (Kangaeru)", "tts_jp": "かんがえる", "id": "Berpikir"},
    {"jp": "走る (Hashiru)", "tts_jp": "はしる", "id": "Berlari"},
    {"jp": "歩く (Aruku)", "tts_jp": "あるく", "id": "Berjalan"},
    {"jp": "入る (Hairu)", "tts_jp": "はいる", "id": "Masuk"},
    {"jp": "出る (Deru)", "tts_jp": "でる", "id": "Keluar"},
    {"jp": "開ける (Akeru)", "tts_jp": "あける", "id": "Membuka"},
    {"jp": "閉める (Shimeru)", "tts_jp": "しめる", "id": "Menutup"},
    {"jp": "洗う (Arau)", "tts_jp": "あらう", "id": "Mencuci"},
    {"jp": "覚える (Oboeru)", "tts_jp": "おぼえる", "id": "Mengingat"},
    {"jp": "忘れる (Wasureru)", "tts_jp": "わすれる", "id": "Lupa"},
    {"jp": "手伝う (Tetsudau)", "tts_jp": "てつだう", "id": "Membantu"},
    {"jp": "約束 (Yakusoku)", "tts_jp": "やくそく", "id": "Janji"},
    {"jp": "準備 (Junbi)", "tts_jp": "じゅんび", "id": "Persiapan"},
    {"jp": "経験 (Keiken)", "tts_jp": "けいけん", "id": "Pengalaman"},
    {"jp": "給料 (Kyuuryou)", "tts_jp": "きゅうりょう", "id": "Gaji"},
    {"jp": "仕事 (Shigoto)", "tts_jp": "しごと", "id": "Pekerjaan"},
    {"jp": "会社 (Kaisha)", "tts_jp": "かいしゃ", "id": "Perusahaan"},
    {"jp": "工場 (Koujou)", "tts_jp": "こうじょう", "id": "Pabrik"},
    {"jp": "安全 (Anzen)", "tts_jp": "あんぜん", "id": "Aman"},
    {"jp": "危険 (Kiken)", "tts_jp": "きけん", "id": "Bahaya"},
    {"jp": "連絡 (Renraku)", "tts_jp": "れんらく", "id": "Hubungan"},
]

# =====================================================================
# 1. PEMINDAI MODEL AI DINAMIS
# =====================================================================
def discover_active_gemini_models(api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    candidates = []
    for m in data.get("models", []):
        name = m.get("name", "")
        methods = m.get("supportedGenerationMethods", [])
        if "generateContent" not in methods:
            continue
        lower = name.lower()
        excluded = ["embedding", "imagen", "veo", "aqa", "tts", "audio", "bisheng"]
        if any(k in lower for k in excluded):
            continue
        if "gemini" in lower:
            candidates.append(name)

    candidates.reverse()
    return candidates


def generate_vocab_via_gemini(used_words):
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY tidak ditemukan.")

    models = discover_active_gemini_models(api_key)
    if not models:
        raise RuntimeError("Tidak ada model aktif ditemukan.")

    avoid_sample = ", ".join(used_words[-150:]) if used_words else "Tidak ada"
    prompt = (
        f"Buat tepat {WORDS_PER_VIDEO} kosakata bahasa Jepang sehari-hari atau kerja (JLPT N5/N4/N3) "
        f"beserta artinya dalam bahasa Indonesia. "
        f"Jangan gunakan kata yang sudah ada di daftar ini: [{avoid_sample}]. "
        f"Format output WAJIB JSON Array murni tanpa markdown, tiap elemen punya key: "
        f"'jp' (contoh: '食べる (Taberu)'), 'tts_jp' (hiragana/katakana murni, contoh: 'たべる'), "
        f"'id' (arti singkat maks 2 kata, contoh: 'Makan')."
    )

    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.7}
    }).encode("utf-8")

    for model_name in models:
        try:
            endpoint = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent?key={api_key}"
            req = urllib.request.Request(
                endpoint, data=payload,
                headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=25) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                raw = result["candidates"][0]["content"]["parts"][0]["text"]
                parsed = extract_json_array(raw)
                if validate_vocab_list(parsed):
                    print(f"[AI Lapis 1] Sukses pakai model: {model_name}")
                    return parsed[:WORDS_PER_VIDEO]
        except Exception as e:
            print(f"[AI Lapis 1] {model_name} dilewati: {e}")
            continue

    raise RuntimeError("Semua model dinamis gagal.")


def generate_vocab_via_public_endpoint(used_words):
    avoid = ", ".join(used_words[-80:]) if used_words else "Tidak ada"
    prompt = (
        f"Berikan {WORDS_PER_VIDEO} kosakata Bahasa Jepang dan artinya dalam Bahasa Indonesia "
        f"selain [{avoid}]. Balas HANYA JSON Array murni berisi objek dengan key "
        f"'jp' (contoh: '食べる (Taberu)'), 'tts_jp' (contoh: 'たべる'), 'id' (contoh: 'Makan')."
    )
    url = f"https://text.pollinations.ai/{urllib.parse.quote(prompt)}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=25) as resp:
        raw = resp.read().decode("utf-8")
        parsed = extract_json_array(raw)
        if validate_vocab_list(parsed):
            print("[AI Lapis 2] Sukses pakai endpoint publik.")
            return parsed[:WORDS_PER_VIDEO]
    raise RuntimeError("Endpoint publik gagal.")


def extract_json_array(text):
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end + 1])
    raise ValueError("JSON Array tidak ditemukan.")


def validate_vocab_list(items):
    if not isinstance(items, list) or len(items) < WORDS_PER_VIDEO:
        return False
    for item in items[:WORDS_PER_VIDEO]:
        if not isinstance(item, dict):
            return False
        if not all(k in item and str(item[k]).strip() for k in ("jp", "tts_jp", "id")):
            return False
    return True


def get_daily_vocabulary():
    used_words = []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            used_words = json.load(f)
    except Exception:
        used_words = []

    try:
        vocab = generate_vocab_via_gemini(used_words)
    except Exception as e1:
        print(f"[Fallback] Lapis 1 gagal ({e1}), ke Lapis 2...")
        try:
            vocab = generate_vocab_via_public_endpoint(used_words)
        except Exception as e2:
            print(f"[Fallback] Lapis 2 gagal ({e2}), ke Lapis 3...")
            unused = [w for w in OFFLINE_VOCAB_BANK if w["jp"] not in used_words]
            if len(unused) < WORDS_PER_VIDEO:
                unused = OFFLINE_VOCAB_BANK[:]
            random.shuffle(unused)
            vocab = unused[:WORDS_PER_VIDEO]

    for v in vocab:
        if v["jp"] not in used_words:
            used_words.append(v["jp"])
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(used_words[-2000:], f, ensure_ascii=False, indent=2)

    return vocab


# =====================================================================
# 2. AUDIO TTS 3 LAPIS
# =====================================================================
def synthesize_speech(text, lang_code, out_mp3):
    # Lapis 1: edge-tts
    try:
        voice = "ja-JP-NanamiNeural" if lang_code == "ja" else "id-ID-GadisNeural"
        subprocess.run(
            ["edge-tts", "--voice", voice, "--text", text, "--write-media", out_mp3],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=25
        )
        if os.path.exists(out_mp3) and os.path.getsize(out_mp3) > 500:
            return
    except Exception as e:
        print(f"[TTS Lapis 1 Gagal - {lang_code}] {e}")

    # Lapis 2: gTTS
    try:
        from gtts import gTTS
        gTTS(text=text, lang=lang_code, slow=False).save(out_mp3)
        if os.path.exists(out_mp3) and os.path.getsize(out_mp3) > 500:
            return
    except Exception as e:
        print(f"[TTS Lapis 2 Gagal - {lang_code}] {e}")

    # Lapis 3: espeak-ng offline
    wav_temp = out_mp3 + ".wav"
    espeak_lang = "ja" if lang_code == "ja" else "id"
    subprocess.run(
        ["espeak-ng", "-v", espeak_lang, "-s", "140", "-w", wav_temp, text],
        check=True
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", wav_temp, "-codec:a", "libmp3lame", "-qscale:a", "2", out_mp3],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    if os.path.exists(wav_temp):
        os.remove(wav_temp)


def create_silence(duration_sec, out_mp3):
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
        "-t", str(duration_sec), "-codec:a", "libmp3lame", "-qscale:a", "2", out_mp3
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# =====================================================================
# 3. RENDER VISUAL
# =====================================================================
def find_best_font(size):
    patterns = [
        "/usr/share/fonts/**/NotoSansCJK*Bold*.ttc",
        "/usr/share/fonts/**/NotoSansCJK*.ttc",
        "/usr/share/fonts/**/NotoSansJP*.otf",
        "/usr/share/fonts/**/NotoSans*.ttf",
        "/usr/share/fonts/**/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/**/*.ttf",
    ]
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        for m in matches:
            try:
                return ImageFont.truetype(m, size)
            except Exception:
                continue
    return ImageFont.load_default()


def render_word_card(jp_display, id_display, output_png):
    img = Image.new("RGB", (WIDTH, HEIGHT), color=BG_COLOR)
    draw = ImageDraw.Draw(img)

    if "(" in jp_display and ")" in jp_display:
        main_jp = jp_display.split("(")[0].strip()
        romaji = "(" + jp_display.split("(", 1)[1].strip()
    else:
        main_jp = jp_display
        romaji = ""

    font_jp = find_best_font(115)
    font_romaji = find_best_font(60)
    font_id = find_best_font(85)

    y_center = HEIGHT // 2

    bbox = draw.textbbox((0, 0), main_jp, font=font_jp)
    draw.text(((WIDTH - (bbox[2] - bbox[0])) / 2, y_center - 240),
              main_jp, font=font_jp, fill=TEXT_COLOR)

    if romaji:
        bbox = draw.textbbox((0, 0), romaji, font=font_romaji)
        draw.text(((WIDTH - (bbox[2] - bbox[0])) / 2, y_center - 80),
                  romaji, font=font_romaji, fill=TEXT_COLOR)

    bbox = draw.textbbox((0, 0), id_display, font=font_id)
    draw.text(((WIDTH - (bbox[2] - bbox[0])) / 2, y_center + 90),
              id_display, font=font_id, fill=TEXT_COLOR)

    img.save(output_png, "PNG")


# =====================================================================
# 4. PERAKITAN VIDEO
# =====================================================================
def build_short_video(vocab_list):
    silence_short = os.path.join(WORK_DIR, "pause_short.mp3")
    silence_gap = os.path.join(WORK_DIR, "pause_gap.mp3")
    create_silence(0.30, silence_short)
    create_silence(0.55, silence_gap)

    segment_files = []

    for idx, item in enumerate(vocab_list):
        jp_text = item["jp"]
        tts_jp = item["tts_jp"]
        id_text = item["id"]

        img_path = os.path.join(WORK_DIR, f"card_{idx}.png")
        render_word_card(jp_text, id_text, img_path)

        jp_audio = os.path.join(WORK_DIR, f"jp_{idx}.mp3")
        id_audio = os.path.join(WORK_DIR, f"id_{idx}.mp3")
        synthesize_speech(tts_jp, "ja", jp_audio)
        synthesize_speech(id_text, "id", id_audio)

        cycle_list = os.path.join(WORK_DIR, f"cycle_{idx}.txt")
        with open(cycle_list, "w", encoding="utf-8") as f:
            for _ in range(REPEATS_PER_WORD):
                f.write(f"file '{os.path.abspath(jp_audio)}'\n")
                f.write(f"file '{os.path.abspath(silence_short)}'\n")
                f.write(f"file '{os.path.abspath(id_audio)}'\n")
                f.write(f"file '{os.path.abspath(silence_gap)}'\n")

        word_audio = os.path.join(WORK_DIR, f"word_audio_{idx}.mp3")
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", cycle_list, "-c", "copy", word_audio
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        segment_mp4 = os.path.join(WORK_DIR, f"segment_{idx}.mp4")
        subprocess.run([
            "ffmpeg", "-y",
            "-loop", "1", "-framerate", str(FPS), "-i", img_path,
            "-i", word_audio,
            "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p",
            "-vf", f"scale={WIDTH}:{HEIGHT}",
            "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
            "-shortest", segment_mp4
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        segment_files.append(segment_mp4)

    master = os.path.join(WORK_DIR, "master.txt")
    with open(master, "w", encoding="utf-8") as f:
        for seg in segment_files:
            f.write(f"file '{os.path.abspath(seg)}'\n")

    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", master, "-c", "copy", OUTPUT_VIDEO
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print(f"[Video Selesai] {OUTPUT_VIDEO}")


# =====================================================================
# 5. UPLOAD YOUTUBE
# =====================================================================
def upload_to_youtube(vocab_list):
    client_id = os.environ.get("YOUTUBE_CLIENT_ID", "").strip()
    client_secret = os.environ.get("YOUTUBE_CLIENT_SECRET", "").strip()
    refresh_token = os.environ.get("YOUTUBE_REFRESH_TOKEN", "").strip()

    if not all([client_id, client_secret, refresh_token]):
        print("[Peringatan] Kredensial YouTube belum lengkap. Skip upload.")
        return

    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    creds = Credentials(
        None, refresh_token=refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=client_id, client_secret=client_secret,
        scopes=["https://www.googleapis.com/auth/youtube.upload"]
    )
    creds.refresh(Request())
    youtube = build("youtube", "v3", credentials=creds)

    first = vocab_list[0]["id"]
    second = vocab_list[1]["id"]
    title = f"Kosakata Bahasa Jepang Harian: {first}, {second} #Shorts #BahasaJepang"
    if len(title) > 95:
        title = title[:95]

    summary = "\n".join([f"• {w['jp']} = {w['id']}" for w in vocab_list])
    description = (
        f"Latihan pendengaran dan hafalan 5 kosakata Bahasa Jepang sehari-hari "
        f"beserta artinya dalam Bahasa Indonesia (diulang 5x per kata).\n\n"
        f"Daftar Kosakata Hari Ini:\n{summary}\n\n"
        f"#Shorts #BelajarBahasaJepang #KosakataBahasaJepang #BahasaJepangPemula "
        f"#JLPTIndonesia #MagangJepang"
    )

    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": [
                "belajar bahasa jepang", "kosakata bahasa jepang",
                "bahasa jepang sehari hari", "arti bahasa jepang",
                "latihan mendengar bahasa jepang", "jlpt n5 indonesia",
                "magang jepang", "tokutei ginou", "shorts bahasa jepang"
            ],
            "categoryId": "27",
            "defaultLanguage": "id",
            "defaultAudioLanguage": "ja"
        },
        "status": {
            "privacyStatus": "public",
            "selfDeclaredMadeForKids": False
        }
    }

    for attempt in range(1, 4):
        try:
            media = MediaFileUpload(OUTPUT_VIDEO, chunksize=-1,
                                    resumable=True, mimetype="video/mp4")
            req = youtube.videos().insert(part="snippet,status",
                                          body=body, media_body=media)
            resp = req.execute()
            print(f"[Upload Sukses] Video ID: {resp.get('id')}")
            return
        except Exception as e:
            print(f"[Upload {attempt} Gagal] {e}")
            time.sleep(10 * attempt)

    raise RuntimeError("Gagal upload setelah 3 percobaan.")


def update_heartbeat():
    with open(HEARTBEAT_FILE, "w", encoding="utf-8") as f:
        f.write(f"Last run: {datetime.now(timezone.utc).isoformat()}\n")


if __name__ == "__main__":
    print("=== MEMULAI AGENT ===")
    daily = get_daily_vocabulary()
    print("Kosakata:", json.dumps(daily, ensure_ascii=False))
    build_short_video(daily)
    upload_to_youtube(daily)
    update_heartbeat()
    print("=== SELESAI ===")
