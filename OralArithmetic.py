import speech_recognition as sr
import re

def is_math_expression(text):
    # Accepts numbers, +, -, *, /, (, ), ., ^, spaces
    pattern = r'^[\d\s\+\-\*\/\(\)\.\^]+$'
    # Remove spaces for stricter check
    text_no_space = text.replace(' ', '')
    return re.match(pattern, text_no_space) is not None

def main():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    print("Say a mathematical expression (e.g., 2 plus 2, 3 * (5 + 1)), or say anything else.")

    while True:
        with mic as source:
            print("Listening... (please speak)")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
        try:
            print("Voice detected, processing...")
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")

            # Convert words to symbols for basic math
            text = text.lower()
            text = text.replace('plus', '+').replace('minus', '-')
            text = text.replace('times', '*').replace('multiplied by', '*')
            text = text.replace('divided by', '/').replace('over', '/')
            text = text.replace('power', '^').replace('to the power of', '^')

            # Remove words like "what is", "calculate", etc.
            text = re.sub(r'(what is|calculate|equals|equal to|result of|evaluate|compute|solve|is|the|by)', '', text)
            text = text.strip()

            if is_math_expression(text):
                print("Mathematical expression detected:", text)
                try:
                    # Evaluate the expression safely
                    result = eval(text.replace('^', '**'))
                    print("Result:", result)
                except Exception as e:
                    print("Could not evaluate expression:", e)
            else:
                print("Not a mathematical expression.")
        except sr.UnknownValueError:
            print("Sorry, could not understand audio. (No voice detected or unclear speech)")
        except sr.RequestError as e:
            print(f"Could not request results; {e}")

        again = input("Press Enter to try again or type 'q' to quit: ")
        if again.strip().lower() == 'q':
            break

if __name__ == "__main__":
    main()