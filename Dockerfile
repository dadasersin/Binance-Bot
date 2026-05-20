FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Default environment variable for the bot to run
ENV BOT_PATH=BINANCE/bot_3.py

# Command to run the bot
CMD ["sh", "-c", "python $BOT_PATH"]
