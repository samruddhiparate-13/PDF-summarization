
import pandas as pd
from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs

model_args = Seq2SeqArgs()

model_args.num_train_epochs = 10
model_args.no_save = True
model_args.evaluate_generated_text = True
model_args.evaluate_during_training = True
model_args.evaluate_during_training_verbose = True
model_args.overwrite_output_dir = True


model = Seq2SeqModel(
    encoder_decoder_type= "bart",
    encoder_decoder_name= "facebook/bart-large",
    args = model_args,
    use_cuda= False
)


data = pd.read_csv(r"D:\downloads\archive\cnn_dailymail\test.csv")


data = data.iloc[0:4, 1:3]


data.columns = ['input_text', 'target_text']


from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(data, test_size=0.2, random_state=42)


model.train_model(train_df, eval_data=val_df)


print(model.predict(['Steven Finn believes he has rediscovered the form that made him one of the most exciting fast-bowling prospects in England. Finn was overlooked for the West Indies tour, but has spent time since the World Cup working on his run-up â€“ and watching videos of his best spells as a reminder of why he became the youngest English bowler to take 50 Test wickets. â€˜Iâ€™ve had my trials and tribulations over the last 12 months,â€™ he said. â€˜But I feel everything at the crease is as I want it to be. Steven Finn believes he\'s regained his previous best form and is ready to push for an England place . Finn admits he\'s \'had my trials and tribulations over the last 12 months\' but he\'s got his \'head straight\' Finn smiles as he helps launchÂ ECB\'s Club Open Days on Tuesday at Brondesbury Cricket Club . â€˜My running style is something Iâ€™ve had my issues with â€“ kneeing the stumps and shortening my run-up, which is what c**ked me up, really. Itâ€™s about trying to get it back to being natural. â€˜Weâ€™ve got footage that I use when Iâ€™ve been bowling at my best. When I was running up as a carefree 21-year-old, I just legged it into the crease. \'It was natural, and I bowled quickly, consistently. My bowlingâ€™s looking pretty similar to that at the moment.â€™ Finnâ€™s claims will be music to the selectorsâ€™ ears as they begin to fret about Englandâ€™s fast-bowling stocks. Finn says he\'s been watching footage of when he was on the top of his form and is returning to that style . Finnâ€™s form return comes at a good time as England'])) 
                    




