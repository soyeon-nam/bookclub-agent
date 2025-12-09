import os
import shutil
import threading
import time
from agent_main import controlled_init,controlled_assistant
from agentstate import BookAgentState
from agent_main import init, assistant
from tools import append_line_to_file
from action import choose_action, action_to_id
from participant import participant_creation, rollback_agents
from ring_buffer import RingBufferWithMetadata
from buffer_manager import TTSManager, STTManager

# This is a simple controlled discussion loop script
# It allows to control when the discussion continues, and to add user messages or rollback the discussion by a number of messages
# Must be use for a controlled discussion when a real play can interact


class DiscussionController:
    def __init__(self):
        self.state: BookAgentState = {
            "messages": [],
            "book": "Cinderella",
            "book_summary": "",
            "participants_name": [],
            "user_name": "choi",
            "topics_hystory": [],
        }

        self.tts_buffer = RingBufferWithMetadata("vr_tts_audio", buffer_size=20 * 1024 * 1024)
        self.stt_buffer = RingBufferWithMetadata("vr_stt_audio", buffer_size=10 * 1024 * 1024)

        self.tts_buffer.create()
        self.stt_buffer.create()

        self.tts_manager = TTSManager(self.tts_buffer)
        self.stt_manager = STTManager(self.stt_buffer)

        self.is_running = False
        self.stt_thread = None
        self.last_tts_time = time.time()
        self.tts_interval = 3.0
        self.user_seq = 0

        self.audio_action = []  #List of [Character_name, audio_file_path, action]
        self.last = 0
        self.last_user = 0
        self.all_participants = list(participant_creation().keys())

    def start(self):
        self.is_running = True

        if os.path.exists("book_club_conversation.txt"):
            os.remove("book_club_conversation.txt")

        self.stt_thread = threading.Thread(target=self._stt_monitoring_loop, daemon=True)
        self.stt_thread.start()

        self._main_logic_loop()

    def _stt_monitoring_loop(self):
        print(">> STT Monitoring Thread Started...")

        while self.is_running:
            result = self.stt_manager.process_stt_buffer()

            if result is None:
                pass
            elif isinstance(result, list):
                self.state["participants_name"] = [self.all_participants[i] for i in result]
            elif isinstance(result, tuple):
                self.last_tts_time = time.time()

                seq, self.user_seq, text = result

                if "end the discussion" in text.lower():
                    self.stop()

                if seq - 1 > len(self.state["messages"]):
                    print("Cannot rollback that many messages.")
                else:
                    removed = self.state["messages"][seq-1:]
                    self.state["messages"] = self.state["messages"][:seq-1]
                    rollback_agents(removed)

                speaker = self.state["user_name"]
                user_message = f"{speaker}: {text}\n"
                self.state["messages"].append(user_message)
                self.tts_manager.sequence_counter = seq
                # self.last_user = 0

                print(self.state["messages"][-2:])
                append_line_to_file("book_club_conversation.txt", user_message)
                print(f"{speaker} spoke. No audio file generated for user messages.")
                print(f"[Input Detected] Seq: {seq}, Text: {text}")

                self.last_tts_time = time.time() - self.tts_interval

            time.sleep(0.01)

    def _main_logic_loop(self):
        print(">> Main Logic Loop Started...")

        self.state = controlled_init(self.state)
        intro_audio_path = os.path.join("dialogue_audio", "President_Introduction.wav")
        self.audio_action.append(["President", intro_audio_path, "Normal 1"])
        if os.path.exists(intro_audio_path):
            self.tts_manager.send_tts_audio(
                audio_file_path=intro_audio_path,
                speaker=255,  # President
                animation_type=0,
                user_sequence=self.user_seq,
                flag=0
            )
        else:
            print(f"Skipping non-existent file: {intro_audio_path}")

        self.last_tts_time = time.time()

        while self.is_running:
            if len(self.state["participants_name"]) != 3:
                time.sleep(0.1)
                continue

            current_time = time.time()

            if current_time - self.last_tts_time > self.tts_interval:
                self._generate_and_send_tts()
                self.last_tts_time = time.time()

            time.sleep(0.1)

    def _generate_and_send_tts(self):
        force = False
        if self.last_user > 5:
            self.last_user = 0
            force = True
        else:
            self.last_user += 1

        self.state, speaker, audio_path = controlled_assistant(self.state, self.last, force)
        action = choose_action(self.state["messages"][-1] if self.state["messages"] else "")
        self.audio_action.append([speaker, audio_path, action])
        print(f"{speaker} spoke. Audio file at: {audio_path}, action chosen: {action}")
        if force:
            print(f"Wait for the participant {self.state['user_name']} to answer the question.")

        if speaker == "President":
            speaker_index = 255
        else:
            speaker_index = self.all_participants.index(speaker)
        if force:
            flag = 1 << 0
        else:
            flag = 0

        if os.path.exists(audio_path):
            self.tts_manager.send_tts_audio(
                audio_file_path=audio_path,
                speaker=speaker_index,
                animation_type=action_to_id(action),
                user_sequence=self.user_seq,
                flag=flag
            )
            time.sleep(1)
        else:
            print(f"Skipping non-existent file: {audio_path}")

        if speaker == "President":
            self.last = 0
        else:
            self.last += 1

    def stop(self):
        self.is_running = False
        shutil.move("book_club_conversation.txt", f"conversation_log/{self.state['user_name']}_ConversationLog.txt")
        if self.stt_thread:
            self.stt_thread.join()

if __name__ == "__main__":
    controller = DiscussionController()
    try:
        controller.start()
    except KeyboardInterrupt:
        controller.stop()
