import telebot
import requests
from ultralytics import YOLO
from config import TOKEN
from functions.stereo_video_detection import detect

bot = telebot.TeleBot(TOKEN)
filepath = "files/"
model = YOLO("yolov8n.pt")


@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, "Howdy, how are you doing?")


@bot.message_handler(content_types=['photo'])
def handle_docs_photo(message):
    photo_id = message.photo[-1].file_id  # получаем id файла
    file_info = bot.get_file(photo_id)  # получаем информацию о файле
    file = requests.get('https://api.telegram.org/file/bot{0}/{1}'.format(TOKEN, file_info.file_path))
    file_name = 'photo' + str(
        message.date) + '.jpg'  # имя для сохраняемого файла
    with open(file_name, 'wb') as f:
        f.write(file.content)  # сохраняем файл на локальном диске

    result = model.predict(source=file_name, show=True)
    bot.reply_to(message, 'фото обработано')


@bot.message_handler(content_types=['video'])
def handle_docs_video(message):
    bot.send_message(message.chat.id, 'Ты отправил мне видео')
    file_info = bot.get_file(message.video.file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    src = "video_telegramm.mp4"
    with open(src, 'wb') as new_file:
        new_file.write(downloaded_file)

    bot.reply_to(message, "Видео скачено, начинаю обработку")
    detect(src, model)

    video = open('output_stereo_video_detected.mp4', 'rb')
    bot.send_message(message.chat.id, 'Видео обработано,отправляю')
    bot.send_video(message.chat.id, video)


# file_info = bot.get_file(message.photo[len(message.photo) - 1].file_id)
# downloaded_file = bot.download_file(file_info.file_path)
# nparr = np.fromstring(downloaded_file, np.uint8)
#
# img_np = cv2.imdecode(nparr, cv2.CV_LOAD_IMAGE_COLOR)
#
# src = filepath + message.photo[0].file_id
# with open(src, 'wb') as new_file:
#     new_file.write(downloaded_file)


# @bot.message_handler(content_types=['video'])
# def handle_docs_audio(message):
#     bot.reply_to(message, 'video обработано')


bot.polling()
