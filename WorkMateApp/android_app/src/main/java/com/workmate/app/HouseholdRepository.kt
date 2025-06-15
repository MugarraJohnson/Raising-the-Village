/**
 * WorkMate App - Data Repository
 * 
 * This repository handles local database operations and server synchronization
 * for household data and vulnerability predictions.
 */

package com.workmate.app

import androidx.room.*
import kotlinx.coroutines.flow.Flow
import retrofit2.Response
import retrofit2.http.*
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Room entity for household data
 */
@Entity(tableName = "households")
data class HouseholdEntity(
    @PrimaryKey val id: String,
    val householdSize: Float,
    val income: Float,
    val age: Float,
    val education: String,
    val progressStatus: String,
    val region: String,
    val programParticipation: String,
    val waterAccess: String,
    val electricityAccess: String,
    val healthcareAccess: String,
    val vulnerabilityLevel: String,
    val confidence: Float,
    val recommendations: String, // JSON string
    val timestamp: Long,
    val isSynced: Boolean = false,
    val syncRetries: Int = 0
)

/**
 * Room DAO for household operations
 */
@Dao
interface HouseholdDao {
    
    @Query("SELECT * FROM households ORDER BY timestamp DESC")
    fun getAllHouseholds(): Flow<List<HouseholdEntity>>
    
    @Query("SELECT * FROM households WHERE isSynced = 0")
    suspend fun getUnsyncedHouseholds(): List<HouseholdEntity>
    
    @Query("SELECT * FROM households WHERE id = :id")
    suspend fun getHouseholdById(id: String): HouseholdEntity?
    
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertHousehold(household: HouseholdEntity)
    
    @Update
    suspend fun updateHousehold(household: HouseholdEntity)
    
    @Query("UPDATE households SET isSynced = 1 WHERE id = :id")
    suspend fun markAsSynced(id: String)
    
    @Query("UPDATE households SET syncRetries = syncRetries + 1 WHERE id = :id")
    suspend fun incrementSyncRetries(id: String)
    
    @Query("DELETE FROM households WHERE id = :id")
    suspend fun deleteHousehold(id: String)
    
    @Query("SELECT COUNT(*) FROM households WHERE isSynced = 0")
    suspend fun getUnsyncedCount(): Int
}

/**
 * Room database
 */
@Database(
    entities = [HouseholdEntity::class],
    version = 1,
    exportSchema = false
)
@TypeConverters(Converters::class)
abstract class WorkMateDatabase : RoomDatabase() {
    abstract fun householdDao(): HouseholdDao
}

/**
 * Type converters for Room database
 */
class Converters {
    @TypeConverter
    fun fromStringList(value: List<String>): String {
        return value.joinToString(",")
    }
    
    @TypeConverter
    fun toStringList(value: String): List<String> {
        return if (value.isEmpty()) emptyList() else value.split(",")
    }
}

/**
 * Network DTOs for API communication
 */
data class HouseholdApiRequest(
    val householdData: HouseholdData,
    val prediction: VulnerabilityPrediction,
    val timestamp: Long,
    val deviceId: String
)

data class HouseholdApiResponse(
    val success: Boolean,
    val message: String,
    val householdId: String?
)

data class SyncResponse(
    val syncedCount: Int,
    val failedCount: Int,
    val message: String
)

data class ModelUpdateResponse(
    val hasUpdate: Boolean,
    val modelVersion: String,
    val downloadUrl: String?,
    val modelSize: Long
)

/**
 * Retrofit API interface
 */
interface WorkMateApiService {
    
    @POST("households")
    suspend fun syncHousehold(@Body request: HouseholdApiRequest): Response<HouseholdApiResponse>
    
    @POST("households/batch")
    suspend fun syncHouseholdsBatch(@Body requests: List<HouseholdApiRequest>): Response<SyncResponse>
    
    @GET("model/version")
    suspend fun checkModelUpdate(@Query("currentVersion") currentVersion: String): Response<ModelUpdateResponse>
    
    @GET("model/download")
    suspend fun downloadModel(@Query("version") version: String): Response<okhttp3.ResponseBody>
}

/**
 * Repository for managing household data and synchronization
 */
@Singleton
class HouseholdRepository @Inject constructor(
    private val localDatabase: WorkMateDatabase,
    private val apiService: WorkMateApiService,
    private val preferencesManager: PreferencesManager
) {
    
    private val householdDao = localDatabase.householdDao()
    
    /**
     * Get all households from local database
     */
    fun getAllHouseholds(): Flow<List<HouseholdEntity>> {
        return householdDao.getAllHouseholds()
    }
    
    /**
     * Save household data locally
     */
    suspend fun saveHouseholdLocally(
        householdData: HouseholdData,
        prediction: VulnerabilityPrediction
    ) {
        val entity = HouseholdEntity(
            id = householdData.householdId.ifEmpty { generateHouseholdId() },
            householdSize = householdData.householdSize,
            income = householdData.income,
            age = householdData.age,
            education = householdData.education,
            progressStatus = householdData.progressStatus,
            region = householdData.region,
            programParticipation = householdData.programParticipation,
            waterAccess = householdData.waterAccess,
            electricityAccess = householdData.electricityAccess,
            healthcareAccess = householdData.healthcareAccess,
            vulnerabilityLevel = prediction.level.name,
            confidence = prediction.confidence,
            recommendations = prediction.recommendations.joinToString("|"),
            timestamp = householdData.timestamp,
            isSynced = false
        )
        
        householdDao.insertHousehold(entity)
    }
    
    /**
     * Sync single household to server
     */
    suspend fun syncToServer(
        householdData: HouseholdData,
        prediction: VulnerabilityPrediction
    ) {
        val request = HouseholdApiRequest(
            householdData = householdData,
            prediction = prediction,
            timestamp = System.currentTimeMillis(),
            deviceId = preferencesManager.getDeviceId()
        )
        
        try {
            val response = apiService.syncHousehold(request)
            if (response.isSuccessful && response.body()?.success == true) {
                // Mark as synced in local database
                householdDao.markAsSynced(householdData.householdId)
            } else {
                throw Exception("Server sync failed: ${response.body()?.message}")
            }
        } catch (e: Exception) {
            // Increment retry count
            householdDao.incrementSyncRetries(householdData.householdId)
            throw e
        }
    }
    
    /**
     * Sync all pending offline data
     */
    suspend fun syncPendingData(): Int {
        val unsyncedHouseholds = householdDao.getUnsyncedHouseholds()
        
        if (unsyncedHouseholds.isEmpty()) return 0
        
        // Convert entities to API requests
        val requests = unsyncedHouseholds.map { entity ->
            HouseholdApiRequest(
                householdData = entity.toHouseholdData(),
                prediction = entity.toVulnerabilityPrediction(),
                timestamp = entity.timestamp,
                deviceId = preferencesManager.getDeviceId()
            )
        }
        
        try {
            // Batch sync for efficiency
            val response = apiService.syncHouseholdsBatch(requests)
            
            if (response.isSuccessful) {
                val syncResponse = response.body()
                if (syncResponse != null) {
                    // Mark successfully synced records
                    unsyncedHouseholds.forEach { entity ->
                        householdDao.markAsSynced(entity.id)
                    }
                    return syncResponse.syncedCount
                }
            }
            
            throw Exception("Batch sync failed")
            
        } catch (e: Exception) {
            // Increment retry counts for failed syncs
            unsyncedHouseholds.forEach { entity ->
                if (entity.syncRetries < 3) { // Max 3 retries
                    householdDao.incrementSyncRetries(entity.id)
                }
            }
            throw e
        }
    }
    
    /**
     * Check for model updates
     */
    suspend fun checkForModelUpdate(): ModelUpdateResponse? {
        return try {
            val currentVersion = preferencesManager.getModelVersion()
            val response = apiService.checkModelUpdate(currentVersion)
            
            if (response.isSuccessful) {
                response.body()
            } else {
                null
            }
        } catch (e: Exception) {
            android.util.Log.w("WorkMate", "Failed to check for model updates", e)
            null
        }
    }
    
    /**
     * Download updated model
     */
    suspend fun downloadModelUpdate(version: String): Boolean {
        return try {
            val response = apiService.downloadModel(version)
            
            if (response.isSuccessful) {
                val body = response.body()
                if (body != null) {
                    // Save model to assets directory
                    val modelBytes = body.bytes()
                    saveModelToAssets(modelBytes, "vulnerability_model_${version}.tflite")
                    
                    // Update version in preferences
                    preferencesManager.setModelVersion(version)
                    
                    true
                } else {
                    false
                }
            } else {
                false
            }
        } catch (e: Exception) {
            android.util.Log.e("WorkMate", "Failed to download model update", e)
            false
        }
    }
    
    /**
     * Get statistics about local data
     */
    suspend fun getDataStatistics(): DataStatistics {
        val totalCount = householdDao.getAllHouseholds().
        val unsyncedCount = householdDao.getUnsyncedCount()
        
        return DataStatistics(
            totalHouseholds = totalCount,
            unsyncedHouseholds = unsyncedCount,
            lastSyncTime = preferencesManager.getLastSyncTime()
        )
    }
    
    /**
     * Helper functions
     */
    private fun generateHouseholdId(): String {
        return "HH_${System.currentTimeMillis()}_${(1000..9999).random()}"
    }
    
    private fun saveModelToAssets(modelBytes: ByteArray, filename: String) {
        // Implementation to save model to internal storage
        // This would typically involve writing to internal app directory
        // and updating the model loading logic to use the new file
    }
    
    /**
     * Extension functions for entity conversion
     */
    private fun HouseholdEntity.toHouseholdData(): HouseholdData {
        return HouseholdData(
            householdId = id,
            householdSize = householdSize,
            income = income,
            age = age,
            education = education,
            progressStatus = progressStatus,
            region = region,
            programParticipation = programParticipation,
            waterAccess = waterAccess,
            electricityAccess = electricityAccess,
            healthcareAccess = healthcareAccess,
            timestamp = timestamp
        )
    }
    
    private fun HouseholdEntity.toVulnerabilityPrediction(): VulnerabilityPrediction {
        val level = VulnerabilityLevel.valueOf(vulnerabilityLevel)
        val recommendationsList = recommendations.split("|")
        
        return VulnerabilityPrediction(
            level = level,
            confidence = confidence,
            probabilities = mapOf(), // Could be reconstructed if needed
            recommendations = recommendationsList
        )
    }
}

/**
 * Data class for repository statistics
 */
data class DataStatistics(
    val totalHouseholds: Int,
    val unsyncedHouseholds: Int,
    val lastSyncTime: Long
)

/**
 * Preferences manager for storing app settings
 */
@Singleton
class PreferencesManager @Inject constructor(
    private val context: android.content.Context
) {
    private val prefs = context.getSharedPreferences("workmate_prefs", android.content.Context.MODE_PRIVATE)
    
    fun getDeviceId(): String {
        return prefs.getString("device_id", null) ?: run {
            val newId = java.util.UUID.randomUUID().toString()
            prefs.edit().putString("device_id", newId).apply()
            newId
        }
    }
    
    fun getModelVersion(): String {
        return prefs.getString("model_version", "1.0.0") ?: "1.0.0"
    }
    
    fun setModelVersion(version: String) {
        prefs.edit().putString("model_version", version).apply()
    }
    
    fun getLastSyncTime(): Long {
        return prefs.getLong("last_sync_time", 0)
    }
    
    fun setLastSyncTime(time: Long) {
        prefs.edit().putLong("last_sync_time", time).apply()
    }
}
