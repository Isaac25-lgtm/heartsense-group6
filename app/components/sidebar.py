import streamlit as st


def render_sidebar():
    """Render the shared sidebar with branding and disclaimer."""
    with st.sidebar:
        st.markdown("## HeartSense")
        st.caption("CSC8204: AI & Machine Learning\n\nGroup 6 | UCU | Easter 2026")
        st.divider()
        st.markdown(
            "**Team**\n"
            "- Nixon Kamugisha (B00321)\n"
            "- Musoke Emmanuel (B31367)\n"
            "- Mwesigwa Simon Peter (B31335)\n"
            "- Omoding Isaac (B31331)"
        )
        st.divider()
        st.warning(
            "This tool is a screening aid for research and demonstration purposes only. "
            "It does not replace clinical diagnosis."
        )
