/**
 * WorkMate App - Background Data Synchronization Worker
 * 
 * This worker handles background synchronization of household data
 * when network connectivity is restored.
 */

package com.workmate.app

import android.content.Context
import androidx.work.*
import androidx.hilt.work.HiltWorker
import dagger.assisted.Assisted
import dagger.assisted.AssistedInject
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Background worker for syncing offline data to server
 */
@HiltWorker
class SyncWorker @AssistedInject constructor(
    @Assisted context: Context,
    @Assisted params: WorkerParameters,
    private val repository: HouseholdRepository,
    private val preferencesManager: PreferencesManager
) : CoroutineWorker(context, params) {

    companion object {
        const val WORK_NAME = "sync_households"
        const val MAX_RETRY_ATTEMPTS = 3
        
        /**
         * Create a unique work request for data synchronization
         */
        fun createSyncRequest(): OneTimeWorkRequest {
            return OneTimeWorkRequestBuilder<SyncWorker>()
                .setConstraints(
                    Constraints.Builder()
                        .setRequiredNetworkType(NetworkType.CONNECTED)
                        .setRequiresBatteryNotLow(true)
                        .build()
                )
                .setBackoffCriteria(
                    BackoffPolicy.EXPONENTIAL,
                    OneTimeWorkRequest.MIN_BACKOFF_MILLIS,
                    java.util.concurrent.TimeUnit.MILLISECONDS
                )
                .addTag(WORK_NAME)
                .build()
        }
        
        /**
         * Schedule periodic sync work
         */
        fun schedulePeriodicSync(context: Context) {
            val periodicSyncRequest = PeriodicWorkRequestBuilder<SyncWorker>(
                15, // Sync every 15 minutes when conditions are met
                java.util.concurrent.TimeUnit.MINUTES
            )
                .setConstraints(
                    Constraints.Builder()
                        .setRequiredNetworkType(NetworkType.CONNECTED)
                        .setRequiresBatteryNotLow(true)
                        .build()
                )
                .addTag(WORK_NAME)
                .build()
            
            WorkManager.getInstance(context).enqueueUniquePeriodicWork(
                WORK_NAME,
                ExistingPeriodicWorkPolicy.KEEP,
                periodicSyncRequest
            )
        }
    }

    override suspend fun doWork(): Result = withContext(Dispatchers.IO) {
        try {
            android.util.Log.d("WorkMate", "Starting background sync...")
            
            // Check if we have internet connectivity
            if (!isNetworkAvailable()) {
                android.util.Log.w("WorkMate", "No network available for sync")
                return@withContext Result.retry()
            }
            
            // Sync pending household data
            val syncedCount = repository.syncPendingData()
            
            // Update last sync time
            preferencesManager.setLastSyncTime(System.currentTimeMillis())
            
            // Check for model updates
            checkForModelUpdates()
            
            android.util.Log.d("WorkMate", "Background sync completed. Synced $syncedCount records")
            
            // Send success notification if any data was synced
            if (syncedCount > 0) {
                showSyncNotification(syncedCount)
            }
            
            Result.success(
                workDataOf(
                    "synced_count" to syncedCount,
                    "sync_time" to System.currentTimeMillis()
                )
            )
            
        } catch (e: Exception) {
            android.util.Log.e("WorkMate", "Sync failed", e)
            
            // Determine if we should retry based on the error type
            when {
                e is java.net.UnknownHostException || 
                e is java.net.SocketTimeoutException -> {
                    // Network issues - retry
                    Result.retry()
                }
                runAttemptCount < MAX_RETRY_ATTEMPTS -> {
                    // Other errors - retry up to max attempts
                    Result.retry()
                }
                else -> {
                    // Max retries reached - fail
                    showSyncErrorNotification(e.message ?: "Unknown error")
                    Result.failure(
                        workDataOf(
                            "error" to e.message,
                            "timestamp" to System.currentTimeMillis()
                        )
                    )
                }
            }
        }
    }
    
    /**
     * Check network availability
     */
    private fun isNetworkAvailable(): Boolean {
        val connectivityManager = applicationContext.getSystemService(Context.CONNECTIVITY_SERVICE) 
            as android.net.ConnectivityManager
        
        return if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.M) {
            val network = connectivityManager.activeNetwork
            val capabilities = connectivityManager.getNetworkCapabilities(network)
            capabilities?.hasCapability(android.net.NetworkCapabilities.NET_CAPABILITY_INTERNET) == true
        } else {
            @Suppress("DEPRECATION")
            connectivityManager.activeNetworkInfo?.isConnected == true
        }
    }
    
    /**
     * Check for and handle model updates
     */
    private suspend fun checkForModelUpdates() {
        try {
            val updateInfo = repository.checkForModelUpdate()
            
            if (updateInfo?.hasUpdate == true) {
                android.util.Log.d("WorkMate", "Model update available: ${updateInfo.modelVersion}")
                
                // Download the updated model
                val success = repository.downloadModelUpdate(updateInfo.modelVersion)
                
                if (success) {
                    android.util.Log.d("WorkMate", "Model updated successfully to version ${updateInfo.modelVersion}")
                    showModelUpdateNotification(updateInfo.modelVersion)
                } else {
                    android.util.Log.w("WorkMate", "Failed to download model update")
                }
            }
        } catch (e: Exception) {
            android.util.Log.w("WorkMate", "Error checking for model updates", e)
        }
    }
    
    /**
     * Show notification for successful sync
     */
    private fun showSyncNotification(syncedCount: Int) {
        val notificationManager = applicationContext.getSystemService(Context.NOTIFICATION_SERVICE) 
            as android.app.NotificationManager
        
        // Create notification channel for Android O and above
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
            val channel = android.app.NotificationChannel(
                "sync_channel",
                "Data Synchronization",
                android.app.NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "Notifications for data synchronization status"
            }
            notificationManager.createNotificationChannel(channel)
        }
        
        val notification = androidx.core.app.NotificationCompat.Builder(applicationContext, "sync_channel")
            .setSmallIcon(android.R.drawable.ic_menu_upload)
            .setContentTitle("WorkMate Sync Complete")
            .setContentText("Successfully synced $syncedCount household records")
            .setPriority(androidx.core.app.NotificationCompat.PRIORITY_LOW)
            .setAutoCancel(true)
            .build()
        
        notificationManager.notify(1001, notification)
    }
    
    /**
     * Show notification for sync errors
     */
    private fun showSyncErrorNotification(error: String) {
        val notificationManager = applicationContext.getSystemService(Context.NOTIFICATION_SERVICE) 
            as android.app.NotificationManager
        
        val notification = androidx.core.app.NotificationCompat.Builder(applicationContext, "sync_channel")
            .setSmallIcon(android.R.drawable.ic_dialog_alert)
            .setContentTitle("WorkMate Sync Failed")
            .setContentText("Data sync failed: $error")
            .setPriority(androidx.core.app.NotificationCompat.PRIORITY_HIGH)
            .setAutoCancel(true)
            .build()
        
        notificationManager.notify(1002, notification)
    }
    
    /**
     * Show notification for model updates
     */
    private fun showModelUpdateNotification(version: String) {
        val notificationManager = applicationContext.getSystemService(Context.NOTIFICATION_SERVICE) 
            as android.app.NotificationManager
        
        val notification = androidx.core.app.NotificationCompat.Builder(applicationContext, "sync_channel")
            .setSmallIcon(android.R.drawable.ic_menu_preferences)
            .setContentTitle("WorkMate Model Updated")
            .setContentText("AI model updated to version $version")
            .setPriority(androidx.core.app.NotificationCompat.PRIORITY_LOW)
            .setAutoCancel(true)
            .build()
        
        notificationManager.notify(1003, notification)
    }
}

/**
 * Network connectivity helper class
 */
class NetworkConnectivityHelper(private val context: Context) {
    
    private val connectivityManager = context.getSystemService(Context.CONNECTIVITY_SERVICE) 
        as android.net.ConnectivityManager
    
    /**
     * Flow that emits network connectivity status
     */
    val isConnected = kotlinx.coroutines.flow.callbackFlow {
        val callback = object : android.net.ConnectivityManager.NetworkCallback() {
            override fun onAvailable(network: android.net.Network) {
                trySend(true)
            }
            
            override fun onLost(network: android.net.Network) {
                trySend(false)
            }
            
            override fun onUnavailable() {
                trySend(false)
            }
        }
        
        // Register callback
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.N) {
            connectivityManager.registerDefaultNetworkCallback(callback)
        } else {
            val request = android.net.NetworkRequest.Builder()
                .addCapability(android.net.NetworkCapabilities.NET_CAPABILITY_INTERNET)
                .build()
            connectivityManager.registerNetworkCallback(request, callback)
        }
        
        // Send initial state
        trySend(isCurrentlyConnected())
        
        // Cleanup when flow is cancelled
        awaitClose {
            connectivityManager.unregisterNetworkCallback(callback)
        }
    }.distinctUntilChanged()
    
    /**
     * Check current connectivity status
     */
    private fun isCurrentlyConnected(): Boolean {
        return if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.M) {
            val network = connectivityManager.activeNetwork
            val capabilities = connectivityManager.getNetworkCapabilities(network)
            capabilities?.hasCapability(android.net.NetworkCapabilities.NET_CAPABILITY_INTERNET) == true
        } else {
            @Suppress("DEPRECATION")
            connectivityManager.activeNetworkInfo?.isConnected == true
        }
    }
}

/**
 * Sync status data class for monitoring
 */
data class SyncStatus(
    val isRunning: Boolean,
    val lastSyncTime: Long,
    val pendingSyncCount: Int,
    val lastError: String?
)

/**
 * Sync manager for coordinating all sync operations
 */
@javax.inject.Singleton
class SyncManager @javax.inject.Inject constructor(
    private val context: Context,
    private val repository: HouseholdRepository,
    private val preferencesManager: PreferencesManager
) {
    
    private val workManager = WorkManager.getInstance(context)
    
    /**
     * Trigger immediate sync
     */
    fun triggerImmediateSync() {
        val syncRequest = SyncWorker.createSyncRequest()
        workManager.enqueueUniqueWork(
            SyncWorker.WORK_NAME,
            ExistingWorkPolicy.REPLACE,
            syncRequest
        )
    }
    
    /**
     * Schedule periodic background sync
     */
    fun schedulePeriodicSync() {
        SyncWorker.schedulePeriodicSync(context)
    }
    
    /**
     * Cancel all sync work
     */
    fun cancelSync() {
        workManager.cancelUniqueWork(SyncWorker.WORK_NAME)
    }
    
    /**
     * Get current sync status
     */
    fun getSyncStatus(): kotlinx.coroutines.flow.Flow<SyncStatus> {
        return kotlinx.coroutines.flow.flow {
            val workInfos = workManager.getWorkInfosByTag(SyncWorker.WORK_NAME).get()
            val isRunning = workInfos.any { it.state == WorkInfo.State.RUNNING }
            
            val pendingCount = try {
                repository.getDataStatistics().unsyncedHouseholds
            } catch (e: Exception) {
                0
            }
            
            emit(SyncStatus(
                isRunning = isRunning,
                lastSyncTime = preferencesManager.getLastSyncTime(),
                pendingSyncCount = pendingCount,
                lastError = null // Could be extracted from work info
            ))
        }
    }
}
