import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import tseslint from 'typescript-eslint'

export default tseslint.config(
  { ignores: ['dist', 'coverage', 'public/mockServiceWorker.js'] },
  {
    extends: [js.configs.recommended, ...tseslint.configs.recommended],
    files: ['**/*.{ts,tsx}'],
    languageOptions: {
      ecmaVersion: 2020,
      globals: globals.browser,
    },
    plugins: {
      'react-hooks': reactHooks,
      'react-refresh': reactRefresh,
    },
    rules: {
      ...reactHooks.configs.recommended.rules,
      'react-refresh/only-export-components': [
        'warn',
        { allowConstantExport: true },
      ],
      // Allow unused vars prefixed with underscore
      '@typescript-eslint/no-unused-vars': [
        'error',
        {
          argsIgnorePattern: '^_',
          varsIgnorePattern: '^_',
          caughtErrorsIgnorePattern: '^_',
        },
      ],
      // Console hygiene (#18): raw console.log/debug/info leak dev-noise and
      // payloads to the production console. Route them through `@/lib/logger`
      // (which no-ops in prod) instead. warn/error are allowed — genuine error
      // reporting must survive production. A `warn` (not `error`) keeps this
      // non-disruptive: it surfaces regressions without breaking the build.
      'no-console': ['warn', { allow: ['warn', 'error'] }],
    },
  },
  // The logger is the single sanctioned console boundary.
  {
    files: ['src/lib/logger.ts'],
    rules: {
      'no-console': 'off',
    },
  },
  // Dev-only tooling. src/mocks/** is the MSW harness whose console.info
  // diagnostics are structurally gated to MODE === 'development' (see
  // initMSW) and can never reach a production browser — they are intentional
  // dev tooling, not production noise.
  {
    files: ['src/mocks/**/*.{ts,tsx}'],
    rules: {
      'no-console': 'off',
    },
  },
  // Relaxed rules for test files (tests legitimately spy on/assert console).
  {
    files: ['**/*.test.{ts,tsx}', '**/*.spec.{ts,tsx}', 'e2e/**/*.ts'],
    rules: {
      '@typescript-eslint/no-explicit-any': 'warn',
      '@typescript-eslint/no-unused-vars': 'warn',
      '@typescript-eslint/no-unsafe-function-type': 'warn',
      '@typescript-eslint/no-this-alias': 'warn',
      'no-console': 'off',
    },
  },
)
