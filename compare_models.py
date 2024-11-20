import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import PeftModel
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model():
    try:
        # Load base model
        base_model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        
        # Configure quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="fp4",
            bnb_4bit_use_double_quant=False,
            bnb_4bit_compute_dtype=torch.float16
        )

        # Load base model with quantization
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # Load your fine-tuned model
        logger.info("Loading fine-tuned model...")
        model = PeftModel.from_pretrained(base_model, "ESE577_chatbot")
        model.eval()

        return model, tokenizer

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def get_response(model, tokenizer, question):
    try:
        # Create pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=256,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15
        )

        # Format prompt
        prompt = f"<s>[INST]Question: {question}[/INST]"
        
        # Generate response
        response = pipe(prompt, max_new_tokens=256, do_sample=True)[0]['generated_text']
        
        # Extract answer part (after [/INST])
        answer = response.split('[/INST]')[-1].strip()
        return answer

    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return f"Error: {str(e)}"

def main():
    print("Loading model and tokenizer...")
    model, tokenizer = load_model()
    print("Model loaded successfully! You can now ask questions.")
    print("Type 'quit' to exit")
    print("-" * 50)

    while True:
        question = input("\nYour question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
            
        if not question:
            print("Please enter a question!")
            continue
            
        print("\nGenerating response...\n")
        response = get_response(model, tokenizer, question)
        print("Response:", response)
        print("-" * 50)

if __name__ == "__main__":
    main()