def mask_rtsp_url(url):
    """
    Masks the password in an RTSP/HTTP URL.
    Format: scheme://user:password@host... -> scheme://user:*****@host...
    Handling passwords with '@' by finding the last '@' before the host.
    """
    if not url:
        return url

    # Find scheme separator
    val = str(url)
    scheme_end = val.find("://")
    if scheme_end == -1:
        return val  # Not a standard URL

    scheme_len = 3  # ://
    start_auth = scheme_end + scheme_len

    # Find end of authority (first / after scheme, or end of string)
    path_start = val.find("/", start_auth)
    if path_start == -1:
        authority = val[start_auth:]
        rest = ""
    else:
        authority = val[start_auth:path_start]
        rest = val[path_start:]

    # In authority, find the LAST '@'
    last_at = authority.rfind("@")
    if last_at == -1:
        return val  # No credentials

    # user:pass is before the last @
    user_pass = authority[:last_at]
    host_port = authority[last_at + 1 :]

    # Split user:pass
    # Standard is first colon separates user from pass
    first_colon = user_pass.find(":")
    if first_colon == -1:
        # No password? e.g. user@host
        return val

    user = user_pass[:first_colon]
    # Password is everything after first colon
    # mask it
    new_authority = f"{user}:*****@{host_port}"

    return f"{val[:start_auth]}{new_authority}{rest}"


def unmask_rtsp_url(new_url, original_url):
    """
    Restores the original password if the new URL contains the placeholder '*****'.
    """
    if not new_url:
        return new_url

    if "*****" not in new_url:
        return new_url

    if not original_url:
        return new_url

    # We need to extract the original password using the exact same logic
    val = str(original_url)
    scheme_end = val.find("://")
    if scheme_end == -1:
        return new_url

    start_auth = scheme_end + 3
    path_start = val.find("/", start_auth)
    if path_start == -1:
        authority = val[start_auth:]
    else:
        authority = val[start_auth:path_start]

    last_at = authority.rfind("@")
    if last_at == -1:
        return new_url

    user_pass = authority[:last_at]
    first_colon = user_pass.find(":")
    if first_colon == -1:
        return new_url

    original_password = user_pass[first_colon + 1 :]

    # Now replace brackets in new_url
    return new_url.replace("*****", original_password, 1)


class TestSecurityMasking:
    def test_mask_rtsp_url_simple(self):
        url = "rtsp://admin:secret123@192.168.1.55:554/stream"
        masked = mask_rtsp_url(url)
        assert masked == "rtsp://admin:*****@192.168.1.55:554/stream"

    def test_mask_rtsp_url_complex_chars(self):
        url = "rtsp://user.name:P@$$w0rd!@10.0.0.1"
        masked = mask_rtsp_url(url)
        assert masked == "rtsp://user.name:*****@10.0.0.1"

    def test_mask_no_credentials(self):
        url = "rtsp://192.168.1.55/stream"
        masked = mask_rtsp_url(url)
        assert masked == "rtsp://192.168.1.55/stream"

    def test_unmask_unchanged(self):
        original = "rtsp://admin:secret123@192.168.1.55:554/stream"
        # User submits marked URL
        submitted = "rtsp://admin:*****@192.168.1.55:554/stream"

        unmasked = unmask_rtsp_url(submitted, original)
        assert unmasked == original

    def test_unmask_changed_ip(self):
        original = "rtsp://admin:secret123@192.168.1.55:554/stream"
        # User changed IP but kept stars
        submitted = "rtsp://admin:*****@192.168.1.99:554/stream"

        expected = "rtsp://admin:secret123@192.168.1.99:554/stream"
        unmasked = unmask_rtsp_url(submitted, original)
        assert unmasked == expected

    def test_unmask_new_password(self):
        original = "rtsp://admin:secret123@192.168.1.55:554/stream"
        # User changed password (no stars)
        submitted = "rtsp://admin:newpass@192.168.1.55:554/stream"

        unmasked = unmask_rtsp_url(submitted, original)
        assert unmasked == submitted
