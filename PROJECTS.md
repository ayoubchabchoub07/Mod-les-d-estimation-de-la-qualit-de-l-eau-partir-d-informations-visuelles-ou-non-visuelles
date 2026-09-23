# Project Portfolio

A technical overview of projects associated with the GitHub account and collaborations provided for `ayoubchabchoub07`.

> **Note:** Strengths, innovation, and architecture are summarized from repository descriptions, README files, and language composition. For repositories without sufficient project documentation, the architecture is marked as an informed interpretation rather than a confirmed implementation detail.

## 1. Water Quality Estimation from Underwater Images

**Repository:** [`ayoubchabchoub07/Mod-les-d-estimation-de-la-qualit-de-l-eau-partir-d-informations-visuelles-ou-non-visuelles`](https://github.com/ayoubchabchoub07/Mod-les-d-estimation-de-la-qualit-de-l-eau-partir-d-informations-visuelles-ou-non-visuelles)

- **Description:** Computer-vision system for estimating water-quality parameters from underwater Tilapia images.
- **Strengths:**
  - Addresses a practical aquaculture and environmental-monitoring problem.
  - Uses clip-level dataset splitting to reduce temporal data leakage.
  - Separates training augmentations from validation and test data.
  - Includes a reproducible preprocessing and dataset-preparation pipeline.
  - Targets multiple parameters: temperature, pH, dissolved oxygen, and turbidity.
- **Technologies used:**
  - Python — 100% of the repository language composition.
  - OpenCV for image processing.
  - NumPy and Pandas for numerical and tabular data processing.
  - Matplotlib for visualization.
  - Scikit-learn for machine-learning utilities.
  - PyTorch, Torchvision, TensorFlow, and `timm` for deep learning and transfer learning.
  - CLAHE, white balance, ImageNet normalization, and image augmentation.
- **Innovation:**
  - Infers physicochemical water properties from visual information rather than relying exclusively on physical sensors.
  - Combines underwater color correction and contrast enhancement with multi-task regression.
  - Uses EfficientNet-B2 as a planned backbone for predicting several water-quality values from the same image.
- **Architecture:**
  - Dataset preparation layer: raw videos/images and metadata are indexed into a structured CSV dataset.
  - Image preprocessing layer: resizing, white balance, CLAHE, normalization, and tensor conversion.
  - Model layer: EfficientNet-B2-based multi-task regression architecture.
  - Evaluation layer: train/validation/test split by video clip, with ablation flags for preprocessing choices.
  - Inference layer: accepts a new underwater image and produces structured JSON predictions.

## 2. Studio / AI Web Application Starter

**Repository:** [`ayoubchabchoub07/studio`](https://github.com/ayoubchabchoub07/studio)

- **Description:** Next.js application created from Firebase Studio with an integrated AI development setup.
- **Strengths:**
  - Modern TypeScript-first web stack.
  - Production-oriented scripts for development, type checking, linting, and builds.
  - Reusable UI component foundation based on Radix UI.
  - Includes Firebase integration and a deployment configuration file.
- **Technologies used:**
  - TypeScript — approximately 98% of the repository language composition.
  - Next.js 15 and React 18.
  - Tailwind CSS and PostCSS.
  - Firebase.
  - Genkit and Google GenAI integration.
  - Radix UI components.
  - React Hook Form, Zod, Recharts, Framer Motion, and Lucide React.
- **Innovation:**
  - Combines a full-stack Next.js application with generative-AI flows through Genkit.
  - Provides a strong base for rapidly prototyping AI-enabled user experiences.
  - Uses composable accessible UI primitives instead of building every interaction from scratch.
- **Architecture:**
  - Next.js application layer for server/client rendering and routing.
  - TypeScript component layer for the user interface.
  - AI layer using Genkit and Google GenAI integrations.
  - Firebase layer for application services and deployment support.
  - Tailwind/Radix presentation layer for styling and reusable interface primitives.

## 3. Mealy Web Meal-Planning Platform

**Repository:** [`ayoubchabchoub07/mealy_projet_web`](https://github.com/ayoubchabchoub07/mealy_projet_web)

- **Description:** AI-assisted web application for recipes, meal planning, nutrition tracking, fridge management, and grocery lists.
- **Strengths:**
  - Covers the complete meal-planning workflow from inventory to shopping.
  - Modular backend routes separate recipes, meal plans, nutrition, grocery lists, users, and fridge data.
  - Includes authentication utilities, Firebase integration, and automated tests.
  - Supports dietary preferences, expiration tracking, and nutrition goals.
- **Technologies used:**
  - Python — approximately 97.2% of the repository language composition.
  - Flask backend with a Blueprint-style modular route structure.
  - Firebase Authentication and Cloud Firestore.
  - React, TypeScript, and Vite frontend.
  - REST APIs, asynchronous JavaScript, HTML5, CSS3, and responsive design.
  - AI services and RAG-related components for recipe recommendations.
- **Innovation:**
  - Connects AI recipe generation with real household context such as fridge contents and dietary preferences.
  - Generates grocery lists from meal plans and can suggest recipes based on available ingredients.
  - Combines meal planning, nutrition monitoring, inventory management, and shopping into one workflow.
- **Architecture:**
  - Frontend/backend monorepo.
  - React + TypeScript + Vite single-page frontend.
  - Flask REST API backend.
  - Route modules expose domain-specific endpoints.
  - Service modules encapsulate AI and RAG functionality.
  - Firebase/Firestore acts as the persistence and authentication platform.
  - Data flow: frontend → REST API → domain routes/services → Firestore and AI services.

## 4. AmenInvest Financial Data Platform

**Repository:** [`Ka3baAnanas/AmenInvest`](https://github.com/Ka3baAnanas/AmenInvest)

- **Description:** Internal platform for collecting, processing, extracting, and exporting financial data from the Tunisian financial market.
- **Strengths:**
  - Handles large-scale document-processing workflows for market publications.
  - Uses background workers so long-running scraping and AI extraction do not block the API.
  - Containerized development and deployment through Docker Compose.
  - Includes progress tracking, retry support, idempotent scraping, and historical data storage.
  - Separates ingestion, extraction, orchestration, storage, and presentation concerns.
- **Technologies used:**
  - Python — approximately 66.3% of the repository language composition.
  - TypeScript and React with Vite for the frontend.
  - FastAPI and Uvicorn for the backend API.
  - Celery and Redis for distributed background tasks.
  - PostgreSQL and Alembic for relational data and migrations.
  - MinIO for S3-compatible object storage and caching.
  - LangChain, Ollama, Camelot, and `pdf2image` for AI-assisted document extraction.
  - Docker and Docker Compose.
- **Innovation:**
  - Automates the conversion of financial publications into structured, reusable financial data.
  - Combines deterministic scraping pipelines with local LLM-assisted extraction.
  - Resolves naming differences between market sources and maintains traceable processing outputs.
  - Supports cached Excel exports and resumable multi-company processing.
- **Architecture:**
  - Microservices-style Docker Compose platform.
  - React/Vite frontend communicates with a FastAPI API gateway.
  - Celery workers process scraping, PDF conversion, and extraction jobs asynchronously.
  - PostgreSQL stores mappings, pipeline logs, and extracted data.
  - MinIO stores document artifacts and generated exports.
  - Specialized scraper and agent modules feed the document-processing pipeline.

## 5. Mazra3ti Il Mabrouka Smart Irrigation System

**Repository:** [`Tarekazabou/mazra3ti_il_mabrouka`](https://github.com/Tarekazabou/mazra3ti_il_mabrouka)

- **Description:** AI-powered smart irrigation platform designed to support women farmers in Tunisia.
- **Strengths:**
  - Provides both automatic AI irrigation and manual valve control.
  - Offers real-time monitoring of soil moisture, temperature, humidity, weather, and irrigation history.
  - Includes separate farmer and administrator interfaces.
  - Uses safety checks to prevent overlapping irrigation operations.
  - Provides Arabic-friendly mobile interaction and voice-assistant support.
- **Technologies used:**
  - HTML, Dart, Python, JavaScript, C++, CMake, and related tooling.
  - Flask backend.
  - Flutter mobile application.
  - Firebase Firestore and Firebase real-time services.
  - XGBoost and Google Gemini API for irrigation decisions.
  - REST APIs and weather-service integration.
- **Innovation:**
  - Applies AI to irrigation decisions using sensor data, crop requirements, weather information, and soil conditions.
  - Gives farmers control through a safe combination of AI automation and manual override.
  - Supports automatic generation of plant features for previously unknown crops.
  - Targets an important social and environmental use case: improving water use for women farmers.
- **Architecture:**
  - Flutter mobile client communicates with a Flask REST backend.
  - Flask routes delegate to farmer, admin, irrigation, valve, weather, and plant services.
  - Firebase provides persistence, real-time state, and event logging.
  - AI decision flow: sensor data → safety/state validation → weather lookup → XGBoost/Gemini decision → Firebase state update and logging.

## 6. Colombus Market Intelligence Stage

**Repository:** [`AhmedZri/colombus-market-intelligence-stage`](https://github.com/AhmedZri/colombus-market-intelligence-stage)

- **Description:** Financial-market back-office platform for identity, counterparties, transactions, liquidity, execution, transaction-cost analysis, market data, and audit operations.
- **Strengths:**
  - Clear modular separation of financial domains.
  - Uses a monorepo to keep the backend and back-office frontend together.
  - Includes database migrations, local PostgreSQL setup, and application tests.
  - Suitable foundation for a data-driven operations and trading platform.
- **Technologies used:**
  - Python — approximately 64.8% of the repository language composition.
  - TypeScript — approximately 34%.
  - Django and Django REST-style backend modules.
  - PostgreSQL with Docker Compose.
  - React, Vite, and Tailwind CSS for the frontend.
- **Innovation:**
  - Organizes a broad financial domain into independently understandable business modules.
  - Provides a foundation for centralized market operations, auditability, and transaction workflows.
  - Combines operational APIs with a dedicated back-office interface.
- **Architecture:**
  - Monorepo with `backend/` and `frontend/` applications.
  - Django backend split into one application per domain.
  - PostgreSQL persistence with migration support.
  - React/Vite/TypeScript/Tailwind administrative client.
  - Frontend communicates with the Django API through a configurable API base URL.

## 7. Colombus Platform

**Repository:** [`seifallahabiriga/colombus-platform`](https://github.com/seifallahabiriga/colombus-platform)

- **Description:** Modular financial-market platform with a Django API backend and React back-office frontend.
- **Strengths:**
  - Domain-driven backend organization.
  - Explicit separation of identity, counterparties, transactions, liquidity, execution, TCA, market data, and audit modules.
  - Uses PostgreSQL and migrations for reliable data evolution.
  - Provides a practical administrative UI foundation.
- **Technologies used:**
  - TypeScript — approximately 58.1% of the repository language composition.
  - Python — approximately 40.4%.
  - Django, PostgreSQL, and Docker Compose.
  - React, Vite, TypeScript, and Tailwind CSS.
  - TailAdmin-based dashboard components and layouts.
- **Innovation:**
  - Translates complex financial operations into a modular platform that can evolve by business capability.
  - Creates a common back-office workspace for multiple market and transaction workflows.
  - Emphasizes maintainability through module ownership and dependency boundaries.
- **Architecture:**
  - Django backend with one application per business module.
  - PostgreSQL database managed through Django migrations.
  - React/Vite frontend for back-office operations.
  - Tailwind-based component and dashboard layer.
  - Docker-supported local development environment.

## 8. Vectors in Orbits — Financially Smart Product Recommender

**Repository:** [`anas-dev0/vectors_in_orbits`](https://github.com/anas-dev0/vectors_in_orbits)

- **Description:** Product recommendation system that combines semantic search with budget and installment constraints.
- **Strengths:**
  - Makes recommendations explainable through semantic, budget, and installment scores.
  - Uses vector search to handle natural-language product queries.
  - Returns both primary recommendations and lower-cost alternatives.
  - Separates product ingestion, retrieval, scoring, API delivery, and frontend presentation.
- **Technologies used:**
  - HTML — approximately 78.6% of the repository language composition.
  - JavaScript, Python, Go, TypeScript, PL/pgSQL, and CSS.
  - Next.js and React frontend.
  - Redux Toolkit for state management.
  - FastAPI and Uvicorn backend.
  - Qdrant vector database.
  - CLIP-ViT-B-32 embeddings through sentence-transformers.
  - Tailwind CSS, Recharts, and a Go web scraper.
- **Innovation:**
  - Personalizes recommendations according to both semantic intent and financial affordability.
  - Models installment fit in addition to product relevance and total price.
  - Provides ranking explanations instead of returning opaque recommendations.
  - Uses vector search to support flexible natural-language product discovery.
- **Architecture:**
  - Next.js/React frontend with Redux state management.
  - FastAPI REST backend with Pydantic validation.
  - Recommendation engine combines Qdrant cosine similarity, budget fit, and installment fit.
  - Qdrant stores 512-dimensional product embeddings and metadata payloads.
  - Go scraper collects product data for ingestion into the vector database.

## 9. Mealy AI-Assisted Meal Planning App

**Repository:** [`Tarekazabou/mealy`](https://github.com/Tarekazabou/mealy)

- **Description:** Cross-platform AI-assisted meal-planning application with recipes, fridge management, nutrition tracking, and grocery planning.
- **Strengths:**
  - Supports Android, iOS, web, Windows, Linux, and macOS through Flutter.
  - Provides a broad feature set across recipes, meal plans, nutrition, groceries, scanning, and user profiles.
  - Uses reusable providers, models, widgets, and services.
  - Includes Firebase authentication and Firestore integration.
- **Technologies used:**
  - Dart — approximately 74.2% of the repository language composition.
  - Python — approximately 16.8%.
  - Flutter and Dart for the mobile/cross-platform client.
  - Flask REST API backend.
  - Firebase Authentication and Cloud Firestore.
  - Google Gemini for AI recipe generation.
  - Provider for Flutter state management.
  - HTTP/Dio, table calendar, Google Fonts, and responsive UI packages.
- **Innovation:**
  - Connects AI-generated recipes to fridge inventory and dietary preferences.
  - Generates grocery lists directly from weekly meal plans.
  - Supports food and receipt scanning as extensions of meal and inventory management.
  - Uses a calendar-based planning experience with AI suggestions.
- **Architecture:**
  - Flutter client organized into models, providers, screens, services, utilities, and reusable widgets.
  - Flask backend organized into domain-specific route modules and shared services.
  - Firebase provides authentication and cloud persistence.
  - API flow: Flutter app → Flask REST endpoints → AI service and Firestore.

## 10. Market Brief Agent

**Repository:** [`TarakKtari/market-brief-Agent`](https://github.com/TarakKtari/market-brief-Agent)

- **Description:** Market-brief and financial-analysis project with a Python-heavy codebase and a JavaScript/CSS/HTML dashboard layer.
- **Strengths:**
  - Python represents approximately 97% of the repository language composition, suggesting a strong data or automation focus.
  - Includes a frontend dashboard area based on JavaScript and web technologies.
  - The repository composition is suitable for combining financial processing with an analyst-facing interface.
- **Technologies used:**
  - Python — approximately 97%.
  - JavaScript — approximately 1.7%.
  - CSS — approximately 1.2%.
  - HTML — approximately 0.1%.
  - Frontend dependencies include common JavaScript parsing and validation tooling.
- **Innovation:**
  - The project appears oriented toward automating or supporting market-brief workflows.
  - Its Python-dominant composition can support data ingestion, analysis, summarization, and report generation.
  - A dashboard layer can make generated market insights accessible to users.
- **Architecture:**
  - **Inferred from available metadata:** Python-centered analysis/automation backend with a web dashboard frontend.
  - Likely separation between data processing, market-brief generation, and browser presentation.
  - The exact framework, data sources, and deployment architecture should be confirmed from the project README and application entry points.

## Portfolio Themes

Across these projects, the strongest recurring themes are:

- **Applied AI:** computer vision, generative AI, recommendation systems, document extraction, and irrigation decisions.
- **Full-stack development:** Python APIs combined with React, Next.js, Flutter, and TypeScript clients.
- **Data-intensive systems:** financial documents, market data, vectors, sensor data, nutrition data, and image datasets.
- **Modular architecture:** domain-specific routes, services, providers, workers, and reusable components.
- **Real-world impact:** environmental monitoring, sustainable agriculture, financial intelligence, and healthier food planning.
- **Production awareness:** authentication, testing, migrations, background processing, containerization, caching, and deployment configuration.
