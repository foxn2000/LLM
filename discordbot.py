import discord
from discord.ext import commands
from discord import app_commands

import torch
from transformers import AutoModelForSequenceClassification
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_path = "DataPilot/ArrowSmart_1.7b_instruction"

model = AutoModelForCausalLM.from_pretrainedmodel = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

torch.cuda.empty_cache()

bot = commands.Bot(command_prefix="!", intents=discord.Intents.all())

def con(in_text):
    text = generator(
        f"ユーザー: {in_text} システム: ",
        max_length = 100,
        do_sample = True,
        temperature = 0.7,
        top_p = 0.9,
        top_k = 0,
        repetition_penalty = 1.1,
        num_beams = 1,
        pad_token_id = tokenizer.pad_token_id,
        num_return_sequences = 1,
        truncation=True
    )
    json_text = text[0]
    result_text = json_text["generated_text"]
    output_text = result_text.replace('ユーザー: '+str(in_text)+' システム:', '')
    return output_text


#起動したときに起こるイベント
@bot.event
async def on_ready():
    print("準備完了")
    await bot.tree.sync()


#!ping
@bot.command()
async def ping(ctx):
    await ctx.send("Pong!")


#/ping
@bot.tree.command(name="ping", description="Ping.")
async def ping(interaction: discord.Interaction):
    await interaction.response.send_message("Pong!")

@bot.tree.command(name="say", description="メッセージをオウム返し")
@app_commands.describe(saying_msg="発言したい文字を入れてください。")
async def say(interaction: discord.Interaction, saying_msg: str):
    await interaction.response.defer()  # ここで処理中であることを伝える
    res = con(saying_msg)
    await interaction.followup.send(f"{res}") 


bot.run("your token")
