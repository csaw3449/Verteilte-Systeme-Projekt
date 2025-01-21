from flask import Flask, render_template, request, redirect, url_for
import boto3
import json
import threading
from datetime import datetime

app = Flask(__name__)

# SQS Configuration
sqs = boto3.resource('sqs', region_name='us-east-1')
html_queue = sqs.get_queue_by_name(QueueName='html_queue')

# Global variables
alarms = []  # List to store alarms
message_id = 1  # Counter for alarm messages
message_id_lock = threading.Lock()  # Thread-safe increment lock


@app.route('/')
def index():
    """
    Fetch alarms from the SQS queue and display them on the index page.
    """
    global message_id  # Access the global variable

    # Fetch messages from the SQS queue
    for message in html_queue.receive_messages(MaxNumberOfMessages=1, WaitTimeSeconds=10):
        try:
            body = json.loads(message.body)  # Parse the message body
            iot_id = body.get('iot_id', 1)  # Extract the `iot_id`
            timestamp_unix = body.get('edge_end2', 'Unknown timestamp')  # Extract the `timestamp`
            timestamp = datetime.utcfromtimestamp(timestamp_unix).strftime('%Y-%m-%d %H:%M:%S')
            alarm_message = {
                "message_id": message_id,
                "iot_id": iot_id,
                "timestamp": timestamp,
                "message": body.get('message', 'Unknown alarm message'),  # Optional: Default message
            }

            # Increment message_id in a thread-safe manner
            with message_id_lock:
                message_id += 1

            alarms.append(alarm_message)
            message.delete()  # Remove the message from the queue after processing

        except Exception as e:
            print(f"Error processing message: {e}", flush=True)

    return render_template('index.html', alarms=alarms)


@app.route('/disarm/<int:message_id>', methods=['POST'])
def disarm_alarm(message_id):
    """
    Remove an alarm with the specified message_id from the alarms list.
    """
    global alarms
    # Filter out the alarm with the matching message_id
    alarms = [alarm for alarm in alarms if alarm["message_id"] != message_id]
    print(f"Alarm {message_id} disarmed!")

    return redirect(url_for('index'))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
