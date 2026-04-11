import torch
import torch.nn as nn

def inspect_sample(text,model, dataloader, vocab, label_names, device, threshold=0.2, idx=0,vectorizer=None):
    
    model.eval()
    
    # Get one batch
    batch = next(iter(dataloader))
    
    # Select sample
    input_ids = batch['input_ids'][idx].unsqueeze(0).to(device)
    attention_mask = batch['attention_mask'][idx].unsqueeze(0).to(device)
    bow = batch['bow'][idx].unsqueeze(0).to(device)
    with torch.no_grad():
        topic_dist, genre_logits, word_dist, _,_ = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bow=bow
        )
    
    # Predictions
    probs = torch.sigmoid(genre_logits)
    preds = (probs > threshold).int()
    
    # Convert to label names
    predicted_labels = [label_names[i] for i in range(len(label_names)) if preds[0][i] == 1]
    # Topic info
    num_topics = topic_dist.size(1)
    # final_linear= model.classifier[-1]
    #combined but we can use it as approximate
    # topic_genre_weights = final_linear.weight[:, :-num_topics]
    topic_distribution = topic_dist[0]
    
    # Top words
    def get_top_words(word_dist, vocab,topic_id):
        # print(f"Word distribution shape: {word_dist.shape}, vocab size: {len(vocab)}")
        beta=torch.softmax(model.beta,dim=1)
        topic_word_dist = beta[topic_id]
        top_indices = topic_word_dist.argsort(descending=True)[:]
        return [vocab[i] for i in top_indices.cpu().numpy()]
    topic_info={}
    plot_data=[]
    top_topics = torch.topk(topic_distribution, k=10)
    for topic_id,score in zip(top_topics.indices, top_topics.values):
        t=topic_id.item()
        words = get_top_words(word_dist, vocab, t)
        score = score.item()
        vectorizer_tokens=set(vectorizer.build_analyzer()(text[0].lower()))
        present_words = [w for w in words if w in vectorizer_tokens]
        print(present_words)
        topic_info[f"Topic {t}"] = {
                    "score": score,
                    "top_words": words[:15],
                    "present_words": list(present_words)
                }
    plot_data.append({
                'topic_info': topic_info
            })
            
    return {
        "predicted_labels": predicted_labels,
        "explain_data": plot_data
    }
