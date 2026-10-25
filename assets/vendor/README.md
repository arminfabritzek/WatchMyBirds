# Bundled browser dependencies

Bootstrap 5.3.2 supplies the existing grid utilities and components. Its CSS and
JavaScript bundle are served by the appliance so a fresh browser can use the UI
without internet access. No Node.js runtime or build step is needed on the Pi.

The files in `bootstrap/5.3.2/` are unmodified upstream release files:

- [bootstrap.min.css](https://raw.githubusercontent.com/twbs/bootstrap/v5.3.2/dist/css/bootstrap.min.css)
- [bootstrap.bundle.min.js](https://raw.githubusercontent.com/twbs/bootstrap/v5.3.2/dist/js/bootstrap.bundle.min.js)
- [LICENSE](https://raw.githubusercontent.com/twbs/bootstrap/v5.3.2/LICENSE) (MIT)

SHA-256:

```text
3017df4a76db5f01c2b99b603d88b03106df13bcfe18e67b7c13c2341d3a67df  bootstrap.min.css
82f64f62bb03c1bc1824b0f9c9e05f70dba33e146818e63cdf5c306c8cf3dedd  bootstrap.bundle.min.js
3d0b0c88216e4752b9afd3d24e36faaf27873b5a45a61458bb3c83e794c26832  LICENSE
```

Source maps are omitted; their relative references matter only when opening
browser developer tools and never contact a CDN. When updating Bootstrap,
replace both assets and the license from the same release, update this directory
and the template paths, and check the UI with external requests blocked.
