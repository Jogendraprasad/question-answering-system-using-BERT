import json
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

def create_pdf(data, output_filename='output.pdf'):
    # Create a PDF document
    pdf_canvas = canvas.Canvas(output_filename, pagesize=letter)

    # Set font and font size
    pdf_canvas.setFont("Helvetica", 12)

    # Counter for numbering questions
    question_number = 1

    # Vertical position for text
    vertical_position = pdf_canvas._pagesize[1] - 50

    # Extract questions and answers
    for section in data['data']:
        for paragraph in section['paragraphs']:
            for qa in paragraph['qas']:
                question = qa['question']
                answer = qa['answers'][0]['text']

                # Add question number, question, and answer to the PDF
                pdf_canvas.drawString(50, vertical_position, f"{question_number}. Question: {question}")
                pdf_canvas.drawString(50, vertical_position - 20, f"Answer: {answer}")

                # Add a newline for separation
                vertical_position -= 40

                # Increment question number for the next iteration
                question_number += 1

                # Move to the next page if there is not enough space
                if vertical_position < 50:
                    pdf_canvas.showPage()
                    vertical_position = pdf_canvas._pagesize[1] - 50

    # Save the PDF
    pdf_canvas.save()

if __name__ == "__main__":
    # Load JSON data
    with open('cdqa_ipc.json', 'r') as json_file:
        data = json.load(json_file)

    # Create PDF
    create_pdf(data, 'output4.pdf')
