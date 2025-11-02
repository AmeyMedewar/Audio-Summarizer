import streamlit as st
import tempfile
import os
from io import BytesIO
import time
import requests
import google.generativeai as genai
from groq import Groq
from langchain_google_genai import ChatGoogleGenerativeAI


# Configure page
st.set_page_config(
    page_title="Voice Transcriber Pro",
    page_icon="🎤",
    layout="wide"
)

# Initialize session state
if 'transcription' not in st.session_state:
    st.session_state.transcription = ""
if 'summary' not in st.session_state:
    st.session_state.summary = ""
if 'groq_client' not in st.session_state:
    st.session_state.groq_client = None
if 'gemini_configured' not in st.session_state:
    st.session_state.gemini_configured = False

def setup_apis(groq_api_key, gemini_api_key):
    """Setup API clients"""
    try:
        # Setup Groq client
        groq_client = Groq(api_key=groq_api_key)
        st.session_state.groq_client = groq_client
        
        # Setup Gemini
        genai.configure(api_key=gemini_api_key)
        st.session_state.gemini_configured = True
        
        return True, "✅ APIs configured successfully!"
    except Exception as e:
        return False, f"❌ Error setting up APIs: {str(e)}"

def transcribe_with_groq(audio_file, groq_client):
    """Transcribe audio using Groq Whisper API"""
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            audio_file.seek(0)
            tmp_file.write(audio_file.read())
            tmp_file_path = tmp_file.name
        
        # Transcribe with Groq
        with open(tmp_file_path, "rb") as file:
            transcription = groq_client.audio.transcriptions.create(
                file=(audio_file.name, file.read()),
                model="whisper-large-v3",
                response_format="text"
            )
        
        # Clean up temp file
        os.unlink(tmp_file_path)
        
        return transcription, None
        
    except Exception as e:
        # Clean up temp file if it exists
        try:
            if 'tmp_file_path' in locals():
                os.unlink(tmp_file_path)
        except:
            pass
        return None, f"Groq transcription error: {str(e)}"

def summarize_with_gemini(text, max_words=150):
    """Summarize text using Gemini API"""
    try:
        model = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
            google_api_key='your key here')
        
        prompt = f"""
        Please provide a concise summary of the following transcribed text. 
        Keep the summary to approximately {max_words} words or less.
        Focus on the main points and key information.
        
        Text to summarize:
        {text}
        
        Summary:
        """
        
        response = model.invoke(prompt)
        return response.content.strip(), None
        
    except Exception as e:
        return None, response

def main():
    st.title("🎤 Voice Transcriber Pro")
    st.markdown("**Powered by Groq Whisper + Google Gemini APIs**")
    
    # API Configuration Sidebar
    with st.sidebar:
        st.header("🔑 API Configuration")
        
        # Groq API Key
        groq_api_key = st.text_input(
            "Groq API Key",
            type="password",
            help="Get your free API key from https://console.groq.com/"
        )
        
        # Gemini API Key
        gemini_api_key = st.text_input(
            "Google Gemini API Key", 
            type="password",
            help="Get your free API key from https://makersuite.google.com/"
        )
        
        # Setup APIs button
        if st.button("🚀 Setup APIs"):
            if groq_api_key and gemini_api_key:
                success, message = setup_apis(groq_api_key, gemini_api_key)
                if success:
                    st.success(message)
                else:
                    st.error(message)
            else:
                st.error("Please enter both API keys!")
        
        # API Status
        st.subheader("📊 API Status")
        if st.session_state.groq_client:
            st.success("✅ Groq API: Ready")
        else:
            st.error("❌ Groq API: Not configured")
            
        if st.session_state.gemini_configured:
            st.success("✅ Gemini API: Ready")
        else:
            st.error("❌ Gemini API: Not configured")
        
        # Summary Settings
        st.subheader("📝 Summary Settings")
        max_summary_words = st.slider("Max Summary Words", 50, 500, 150)
        
        # API Info
        st.subheader("ℹ️ API Information")
        st.markdown("""
        **Groq API (Transcription)**:
        - Free tier: 14,400 requests/day
        - Very fast Whisper processing
        - Get key: https://console.groq.com/
        
        **Gemini API (Summarization)**:
        - Free tier: 15 req/min, 1500 req/day
        - High-quality summarization
        - Get key: https://makersuite.google.com/
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📁 Upload Audio File")
        
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['mp3', 'wav', 'm4a', 'flac', 'mp4', 'mpeg', 'mpga', 'webm'],
            help="Supports most audio formats. Max size: 25MB for Groq API"
        )
        
        if uploaded_file is not None:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            
            # File info
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # MB
            st.info(f"📏 File size: {file_size:.2f} MB")
            
            if file_size > 25:
                st.warning("⚠️ File is larger than 25MB. Groq API might reject it.")
            
            # Audio player
            st.audio(uploaded_file.getvalue())
    
    with col2:
        st.header("🔄 Processing")
        
        # Check if APIs are configured
        apis_ready = st.session_state.groq_client and st.session_state.gemini_configured
        
        if not apis_ready:
            st.warning("⚠️ Please configure your API keys in the sidebar first!")
        
        if uploaded_file is not None and apis_ready:
            # Transcribe button
            if st.button("🎯 Transcribe with Groq", type="primary", disabled=not apis_ready):
                with st.spinner("🎤 Transcribing with Groq Whisper... Usually takes 5-30 seconds."):
                    start_time = time.time()
                    
                    transcription, error = transcribe_with_groq(uploaded_file, st.session_state.groq_client)
                    
                    end_time = time.time()
                    
                if transcription:
                    st.session_state.transcription = transcription
                    st.success(f"✅ Transcription completed in {end_time - start_time:.2f} seconds!")
                    st.balloons()
                else:
                    st.error(error)
            
            # Summarize button (only show if transcription exists)
            if st.session_state.transcription:
                if st.button("📝 Summarize with Gemini", disabled=not apis_ready):
                    with st.spinner("🤖 Generating summary with Gemini..."):
                        start_time = time.time()
                        
                        summary, error = summarize_with_gemini(
                            st.session_state.transcription,
                            max_summary_words
                        )
                        
                        end_time = time.time()
                    
                    if summary:
                        st.session_state.summary = summary
                        st.success(f"✅ Summary generated in {end_time - start_time:.2f} seconds!")
                    else:
                        st.error(error)
        elif uploaded_file is not None:
            st.info("👆 Configure your API keys first!")
    
    # Results section
    if st.session_state.transcription or st.session_state.summary:
        st.header("📊 Results")
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["📝 Transcription", "📋 Summary", "📈 Statistics"])
        
        with tab1:
            if st.session_state.transcription:
                st.subheader("🎤 Transcribed Text")
                st.text_area(
                    "Full transcription from Groq Whisper:",
                    value=st.session_state.transcription,
                    height=300,
                    key="transcription_display"
                )
                
                # Download transcription
                st.download_button(
                    label="📥 Download Transcription",
                    data=st.session_state.transcription,
                    file_name=f"transcription_{uploaded_file.name if uploaded_file else 'audio'}.txt",
                    mime="text/plain"
                )
            else:
                st.info("No transcription available yet. Upload an audio file and click 'Transcribe with Groq'.")
        
        with tab2:
            if st.session_state.summary:
                st.subheader("🤖 AI-Generated Summary")
                st.text_area(
                    "Summary from Google Gemini:",
                    value=st.session_state.summary,
                    height=200,
                    key="summary_display"
                )
                
                # Download summary
                st.download_button(
                    label="📥 Download Summary",
                    data=st.session_state.summary,
                    file_name=f"summary_{uploaded_file.name if uploaded_file else 'audio'}.txt",
                    mime="text/plain"
                )
                
                # Re-summarize button
                if st.button("🔄 Re-generate Summary"):
                    with st.spinner("🤖 Re-generating with current settings..."):
                        summary, error = summarize_with_gemini(
                            st.session_state.transcription,
                            max_summary_words
                        )
                    
                    if summary:
                        st.session_state.summary = summary
                        st.rerun()
                    else:
                        st.error(error)
            else:
                st.info("No summary available yet. Generate a transcription first, then click 'Summarize with Gemini'.")
        
        with tab3:
            if st.session_state.transcription:
                # Statistics
                transcription_words = len(st.session_state.transcription.split())
                transcription_chars = len(st.session_state.transcription)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📝 Transcription Words", transcription_words)
                    st.metric("🔤 Transcription Characters", transcription_chars)
                
                if st.session_state.summary:
                    summary_words = len(st.session_state.summary.split())
                    summary_chars = len(st.session_state.summary)
                    compression_ratio = (summary_words / transcription_words) * 100 if transcription_words > 0 else 0
                    
                    with col2:
                        st.metric("📋 Summary Words", summary_words)
                        st.metric("🔤 Summary Characters", summary_chars)
                    
                    with col3:
                        st.metric(
                            "📊 Compression Ratio",
                            f"{compression_ratio:.1f}%",
                            help="Percentage of original text retained in summary"
                        )
                        
                        # Reading time estimate
                        reading_time_orig = max(1, transcription_words // 200)  # 200 WPM
                        reading_time_summary = max(1, summary_words // 200)
                        st.metric(
                            "⏱️ Time Saved",
                            f"{reading_time_orig - reading_time_summary} min",
                            help=f"Original: ~{reading_time_orig}min, Summary: ~{reading_time_summary}min"
                        )
            else:
                st.info("Statistics will appear after transcription.")
    
    # Clear results
    if st.session_state.transcription or st.session_state.summary:
        st.header("🗑️ Reset")
        if st.button("Clear All Results", type="secondary"):
            st.session_state.transcription = ""
            st.session_state.summary = ""
            st.rerun()
    
    # Instructions
    with st.expander("📖 Setup Instructions"):
        st.markdown("""
        ### 🚀 How to get API keys:
        
        **1. Groq API Key (Free)**:
        - Go to https://console.groq.com/
        - Sign up for free account
        - Go to API Keys section
        - Create new API key
        - Free tier: 14,400 requests/day
        
        **2. Google Gemini API Key (Free)**:
        - Go to https://makersuite.google.com/
        - Sign in with Google account
        - Click "Get API Key"
        - Create new API key
        - Free tier: 15 requests/minute, 1500/day
        
        ### 💡 Usage Tips:
        - Groq transcription is very fast (5-30 seconds)
        - Gemini provides high-quality summaries
        - Both services have generous free tiers
        - No local processing - works on any device
        - Audio files up to 25MB supported
        
        ### 🔒 Privacy:
        - Audio is sent to Groq for transcription
        - Text is sent to Google for summarization
        - Check respective privacy policies for details
        """)

if __name__ == "__main__":

    main()
