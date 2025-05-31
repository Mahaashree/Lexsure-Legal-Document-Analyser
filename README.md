#Lexsure - Legal Document Risk Analyzer

A  FastAPI-based application that analyzes legal documents for potential risks and generates annotated PDFs with highlighted risk clauses and suggested improvements.

## Features

- **Intelligent Risk Detection**: Uses advanced LLM analysis to identify potential legal risks in contracts and agreements
- **Visual Annotations**: Generates highlighted PDFs with color-coded risk levels and interactive annotations
- **Contextual Analysis**: Customizable analysis based on document type, jurisdiction, and business context
- **Smart Amendments**: AI-powered suggestions for improving risky clauses
- **Scalable Processing**: Handles large documents with optimized chunking and parallel processing
- **RESTful API**: Clean API endpoints for integration with other systems

## 🛠 Installation

### Prerequisites

- Python 3.8+
- Together AI API key
- Git

### Local Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd legal-document-analyzer
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

## ⚙️ Configuration

Create a `.env` file in the project root:

```env
# Required
TOGETHER_API_KEY=your_together_api_key_here

# Optional
UPLOAD_DIR=uploads
MAX_FILE_SIZE=10485760  # 10MB in bytes
SIMILARITY_THRESHOLD=0.55
MAX_PAGES_TO_PROCESS=50
```

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `TOGETHER_API_KEY` | Together AI API key for LLM analysis | - | ✅ |
| `UPLOAD_DIR` | Directory for temporary file storage | `uploads` | ❌ |
| `MAX_FILE_SIZE` | Maximum upload size in bytes | `10485760` (10MB) | ❌ |
| `SIMILARITY_THRESHOLD` | Text matching sensitivity (0.0-1.0) | `0.55` | ❌ |
| `MAX_PAGES_TO_PROCESS` | Maximum pages to analyze per document | `50` | ❌ |

## 🚀 Usage

### Starting the Server

```bash
# Development
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Web Interface

1. Navigate to `http://localhost:8000`
2. Upload a PDF legal document
3. Configure analysis parameters:
   - Document Type (Contract, Agreement, etc.)
   - Jurisdiction (United States, EU, etc.)
   - Business Type
   - Specific concerns (liability, termination, etc.)
4. Click "Analyze Document"
5. Download the annotated PDF with risk highlights



### Interactive API Documentation

Visit `http://localhost:8000/docs` for Swagger UI documentation.

## 🔍 Features in Detail

### Risk Analysis Categories

The system identifies risks in the following categories:

- **Liability**: Exposure to legal responsibility
- **Termination**: Contract ending conditions
- **Payment**: Financial obligations and terms
- **Confidentiality**: Information protection clauses
- **Intellectual Property**: IP rights and restrictions
- **Dispute Resolution**: Conflict handling mechanisms
- **Indemnification**: Protection from losses
- **Limitation of Liability**: Caps on damages

### Risk Levels

- 🔴 **High**: Critical risks requiring immediate attention
- 🟠 **Medium**: Important risks that should be reviewed
- 🟡 **Low**: Minor risks worth noting

### Annotation Features

- **Color-coded highlights** based on risk level
- **Interactive popups** with detailed risk explanations
- **Amendment suggestions** for improving problematic clauses
- **Contextual analysis** based on document type and jurisdiction

### Processing Optimizations

- **Intelligent chunking**: Breaks large documents into manageable pieces
- **Parallel processing**: Analyzes multiple sections simultaneously
- **Duplicate detection**: Removes similar risk clauses
- **Memory management**: Efficient handling of large PDFs
- **Error recovery**: Graceful handling of processing failures

##  Architecture

### Technology Stack

- **Backend**: FastAPI (Python)
- **PDF Processing**: PyMuPDF (fitz)
- **LLM Integration**: Together AI
- **Text Analysis**: difflib, regex
- **File Handling**: tempfile, pathlib
- **Async Processing**: asyncio, concurrent.futures

### Project Structure

```
legal-document-analyzer/
├── main.py                 # Main FastAPI application
├── requirements.txt        # Python dependencies
├── .env.example           # Environment template
├── README.md              # This file
├── static/                # Static assets
├── templates/             # HTML templates
│   └── index.html         # Upload interface
├── uploads/               # Temporary file storage
└── logs/                  # Application logs
```

### Data Flow

1. **Upload**: User uploads PDF via web interface or API
2. **Validation**: File size, type, and content validation
3. **Extraction**: Text extraction page by page using PyMuPDF
4. **Chunking**: Large documents split into analysis chunks
5. **Analysis**: LLM analyzes each chunk for risk patterns
6. **Deduplication**: Similar risks are merged
7. **Amendment Generation**: AI creates improvement suggestions
8. **Annotation**: PDF is annotated with highlights and popups
9. **Response**: Annotated PDF returned to user

### Security Considerations

- File size limits prevent DoS attacks
- Temporary files are automatically cleaned up
- Input validation on all endpoints
- CORS middleware for cross-origin requests
- No permanent storage of user documents


### Adding New Risk Categories

1. Update the `category` enum in the LLM prompts
2. Add category-specific analysis logic
3. Update the color mapping in `create_annotation()`
4. Add tests for the new category

## 📊 Performance

### Benchmarks

- **Small documents** (1-5 pages): ~30-60 seconds
- **Medium documents** (6-20 pages): ~1-3 minutes
- **Large documents** (21-50 pages): ~2-5 minutes

### Scaling Considerations

- **Horizontal scaling**: Deploy multiple instances behind a load balancer
- **Caching**: Implement Redis for repeated document analysis
- **Queue system**: Use Celery for background processing
- **Database**: Add PostgreSQL for user management and document history

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation for API changes
- Use type hints for all functions
- Add logging for debugging purposes

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Email**: mahaashreeofficial@gmail.com

## 🎯 Roadmap

- [ ] Support for Word documents (.docx)
- [ ] Multi-language analysis support
- [ ] User authentication and document history
- [ ] Batch processing API
- [ ] Integration with popular contract management systems
- [ ] Advanced analytics dashboard
- [ ] Custom risk category definitions
- [ ] Webhook notifications for completed analyses

---

Built with ❤️ using FastAPI and Together AI
