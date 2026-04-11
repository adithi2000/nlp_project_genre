import streamlit as st
from combined import do_inference


st.title("Genre Prediction and Topic Explanation")
input_text = st.text_area("Enter text for genre prediction:", height=200)
threshold = st.slider("Prediction Threshold", 0.0, 1.0, 0.2, 0.01)
#check box for true labels # just for user reference
true_labels_input = st.text_input("For your reference, enter true genres (comma-separated):")
true_labels = [label.strip() for label in true_labels_input.split(",")] if true_labels_input else None
if st.button("Predict"):
    if input_text.strip():
        results = do_inference([input_text], threshold=threshold, true_labels=true_labels )
        st.subheader("Predicted Genres:")
        st.write(", ".join(results["predicted_labels"]))
        
        st.subheader("Topic Explanation:")
        st.markdown("**Topics:**")
        st.markdown(len(results['explain_data']))
        # for i in results['explain_data']:
        #     st.markdown("---")
        #     topic_info = i
        #     print(topic_info)
        topic_info=results['explain_data'][0]['topic_info']
        # st.json(topic_info)
        for topic,details in topic_info.items():
            # st.markdown(topic)
            # st.markdown(f"Score: {details['score']}")
            # st.markdown(f"top words: {details['top_words']}")
            # st.markdown(f"present summary word order: {details['present_words']}")
            with st.expander(f"{topic}| Score: {details['score']:.4f}"):
                st.write("**Top words:**")
                st.write(",".join(details['top_words']))
                st.write("**Present word order for the topic**")
                st.write(",".join(details['present_words']))
                             
    else:
        st.warning("Please enter some text for prediction.")

