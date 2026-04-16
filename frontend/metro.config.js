const { getDefaultConfig, mergeConfig } = require('@react-native/metro-config');

/**
 * Metro configuration
 * https://reactnative.dev/docs/metro
 *
 * @type {import('@react-native/metro-config').MetroConfig}
 */
const config = {
  transformer: {
    // Use worker threads instead of child processes to avoid Windows EPERM spawn failures.
    unstable_workerThreads: true,
  },
};

module.exports = mergeConfig(getDefaultConfig(__dirname), config);
