# Install the R packages needed by titanic.R, with pinned versions.
#
# The repository below is a dated snapshot of CRAN (from Posit Package
# Manager). Installing from a fixed date means every build gets exactly
# the same package versions, the R equivalent of pinning versions in
# requirements.txt.
options(repos = c(CRAN = "https://packagemanager.posit.co/cran/2024-10-30"))

# Package name = expected version
packages <- c(dplyr = "1.1.4")

install.packages(names(packages))

# Stop the Docker build with a clear error if a package is missing or
# has an unexpected version
for (pkg in names(packages)) {
  installed <- as.character(packageVersion(pkg))
  if (installed != packages[[pkg]]) {
    stop(sprintf("%s: expected version %s, got %s",
                 pkg, packages[[pkg]], installed))
  }
  cat(sprintf("Installed %s %s\n", pkg, installed))
}