# MshIO
# License: Apache-2.0

if(TARGET mshio)
    return()
endif()

message(STATUS "Third-party: creating target 'mshio'")


include(FetchContent)
FetchContent_Declare(
    mshio
    GIT_REPOSITORY https://github.com/qnzhou/MshIO.git
    GIT_TAG 45a8aa713f8ed6c321ad51a786b84c7bb0e50248
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(mshio)