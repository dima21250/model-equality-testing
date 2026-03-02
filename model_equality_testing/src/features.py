import numpy as np

def decode_sample_to_strings(sample):
    """
    Decode a sample's completion_sample (unicode codepoints) to strings.

    Args:
        sample: Sample object with completion_sample attribute containing unicode codepoints

    Returns:
        List of decoded strings
    """
    decoded_strings = []
    for i in range(sample.N):
        codepoints = sample.completion_sample[i].tolist()
        valid_codepoints = [c for c in codepoints if c != -1]
        text = "".join([chr(c) for c in valid_codepoints])
        decoded_strings.append(text)
    return decoded_strings

def get_vader_scores(sample):
    """
    Layman: This turns a pile of text into a pile of numbers (sentiment).
    """
    # Assuming a hypothetical vader implementation
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()

    # Extract strings from the numeric completion sample
    # (Simplified: assumes you've decoded the unicode codepoints back to strings)
    decoded_strings = decode_sample_to_strings(sample)
    return np.array([analyzer.polarity_scores(s)['compound'] for s in decoded_strings])
