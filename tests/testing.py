from style_bert_vits2.nlp.japanese import g2p
from style_bert_vits2.nlp.japanese.bert_feature import extract_bert_feature
from style_bert_vits2.nlp.japanese.normalizer import normalize_text
from style_bert_vits2.nlp.japanese.g2p import text_to_sep_kata
import torch

if __name__ == "__main__":
    # Example Japanese sentence
    #text = "彼が選んだ。"
    text = "テネシー大学,デューク大学,フロリダ大学などからのオファーもある中,彼が選んだのはノートルダム大学であった."
    #text="九月五日,シカゴ,ベアーズとプラクティス,スクワッドとして契約を結んだ."
    norm_text = normalize_text(text)
    phones, tones, word2ph = g2p.g2p(norm_text, use_jp_extra=True, raise_yomi_error=False)
    # Get the joined text used in extract_bert_feature
    bert_text = "".join(text_to_sep_kata(norm_text, raise_yomi_error=False)[0])
    print("Original text:", text)
    print("Normalized text:", norm_text)
    print("BERT text:", bert_text)
    print("Phones:", phones)
    print("Tones:", tones)
    print("word2ph:", word2ph)
    print("len(bert_text):", len(bert_text))
    print("len(word2ph):", len(word2ph))
    print("len(phones):", len(phones))

    # Run extract_bert_feature using normalized text
    device = "cpu"
    bert_feature = extract_bert_feature(bert_text, word2ph, device)
    print("BERT feature shape:", bert_feature.shape)
    print("BERT feature (first 2 rows):\n", bert_feature[:2])
