from style_bert_vits2.nlp.japanese import g2p
from style_bert_vits2.nlp.japanese import g2p_utils
from style_bert_vits2.nlp.japanese import normalizer
from style_bert_vits2.nlp.japanese.bert_feature import extract_bert_feature
from style_bert_vits2.nlp.japanese.normalizer import normalize_text
from style_bert_vits2.nlp.japanese.g2p import text_to_sep_kata
import torch
import json
import re 

if __name__ == "__main__":
    # Example Japanese sentence
    #text = "彼が選んだ。"
    #text = "テネシー大学,デューク大学,フロリダ大学などからのオファーもある中,彼が選んだのはノートルダム大学であった."
    #text="九月五日,シカゴ,ベアーズとプラクティス,スクワッドとして契約を結んだ."
    #text="十二月十九日,クリス,コンテが故障者リスト入りするのと入れ違いにアクティブロースター入りした."
    #text = "本シーズンのプリメーラ,ディビシオンは,千九百九十六-千九百九十七シーズンまで暫定的に二チーム増やして全二十二チーム体制で運営されることになった."
    #norm_text = normalize_text(text)
    #phones, tones, word2ph = g2p.g2p(norm_text, use_jp_extra=True, raise_yomi_error=False)
    # Get the joined text used in extract_bert_feature
    #bert_text = "".join(text_to_sep_kata(norm_text, raise_yomi_error=False)[0])
    #print("Original text:", text)
    #print("Normalized text:", norm_text)
    #print("BERT text:", bert_text)
    #print("Phones:", phones)
    #print("Tones:", tones)
    #print("word2ph:", word2ph)
    #print("len(bert_text):", len(bert_text))
    #print("len(word2ph):", len(word2ph))
    #print("len(phones):", len(phones))

    # Run extract_bert_feature using normalized text
    #device = "cpu"
    #bert_feature = extract_bert_feature(bert_text, word2ph, device)
    #print("BERT feature shape:", bert_feature.shape)
    #print("BERT feature (first 2 rows):\n", bert_feature[:2])

    # kata stuff
    #text = "こんにちは、初めまして。あなたの名前はなんていうの？ Do you also want me to try speaking English? いやだったら、日本語しゃべようか？"
    text = "How about english at the start. こんにちは、初めまして。あなたの名前はなんていうの？ Do you also want me to try speaking English? いやだったら、日本語しゃべようか？"
    def split_jp_en_blocks(text):
        # Japanese block: Hiragana, Katakana, Kanji, prolonged sound mark, punctuation
        jp = r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBFー。、！？\uFF01-\uFF0F]+'
        # English block: a-zA-Z, spaces, and attached punctuation
        en = r'[a-zA-Z\s]+[.,!?\'"]*'
        pattern = re.compile(f'({jp})|({en})')
        blocks = []
        pos = 0
        for m in pattern.finditer(text):
            if m.start() > pos:
                # Add any intervening non-matched text (e.g. symbols)
                blocks.append(text[pos:m.start()])
            blocks.append(m.group(0))
            pos = m.end()
        if pos < len(text):
            blocks.append(text[pos:])
        # Remove empty strings and strip leading/trailing spaces
        blocks = [b.strip() for b in blocks if b.strip()]
        return blocks


    pieces = split_jp_en_blocks(text)
    print(pieces)

    norm_text = normalize_text(text)
    kata_tone = g2p_utils.g2kata_tone(norm_text)
    kata_tone_json_str = json.dumps(kata_tone, ensure_ascii=False)
    print("Original text:", text)
    print("Normalized text:", norm_text)
    print("Kata tone:", kata_tone_json_str)
    print("len(norm_text):", len(norm_text))
    print("len(kata_tone):", len(kata_tone))
    print("kata json", kata_tone_json_str)