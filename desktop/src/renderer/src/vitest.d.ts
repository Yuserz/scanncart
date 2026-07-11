// Brings the @testing-library/jest-dom matcher augmentations (toHaveTextContent,
// toBeInTheDocument, …) into the renderer TS program so `tsc` typechecks tests.
import '@testing-library/jest-dom/vitest'
